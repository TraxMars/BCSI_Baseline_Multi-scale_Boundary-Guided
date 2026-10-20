"""Four encoder boundary heads using a shared original-resolution BFA target.

This is the labeled-only boundary component of BFA, adapted to unchanged MT.
No unlabeled LoG or extra segmentation path is included. Gradient routing is
external: deep and local modes therefore share exactly this forward and loss.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from utils.boundary_supervision import binary_boundary_loss


def _foreground(labels):
    if labels.ndim == 4:
        labels = labels.unsqueeze(1)
    if labels.ndim != 5 or labels.shape[1] != 1 or labels.shape[0] == 0:
        raise ValueError('Expected nonempty labels [N,D,H,W] or [N,1,D,H,W]')
    if not torch.isfinite(labels).all():
        raise ValueError('Boundary labels must be finite')
    return (labels > 0).to(dtype=torch.float32)


class LaplacianBoundary(nn.Module):
    """BFA foreground-inner 26-neighbor boundary; zero padding is reference.

    Replicate is an explicit diagnostic alternative. Zero padding invents
    outside background when a foreground object touches the crop faces.
    """
    def __init__(self, padding_mode='zero'):
        super().__init__()
        if padding_mode not in ('zero', 'replicate'):
            raise ValueError('padding_mode must be zero or replicate')
        self.padding_mode = padding_mode
        kernel = -torch.ones(1, 1, 3, 3, 3, dtype=torch.float32)
        kernel[0, 0, 1, 1, 1] = 26
        self.register_buffer('kernel', kernel)

    @torch.no_grad()
    def forward(self, labels):
        mask = _foreground(labels)
        kernel = self.kernel.to(device=mask.device, dtype=mask.dtype)
        if self.padding_mode == 'replicate':
            mask = F.pad(mask, (1, 1, 1, 1, 1, 1), mode='replicate')
            response = F.conv3d(mask, kernel)
        else:
            response = F.conv3d(mask, kernel, padding=1)
        return (response.clamp(min=0) > 0.1).to(dtype=torch.float32)


class _BFAHead(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(channels, 16, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm3d(16)
        self.relu = nn.ReLU()
        self.conv_out = nn.Conv3d(16, 1, 1, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, a=1)
        nn.init.kaiming_normal_(self.conv_out.weight, a=1)

    def forward(self, feature):
        return self.conv_out(self.relu(self.bn(self.conv(feature))))


@dataclass
class BFABoundaryResult:
    features: list
    logits: list
    upsampled_logits: list
    target: torch.Tensor
    targets: list
    losses: list
    diagnostics: list

    @property
    def total_loss(self):
        return torch.stack(self.losses).mean()


class BFAEncoderBoundarySupervision(nn.Module):
    """Independent x1..x4 heads; only labeled features enter each head BN."""
    def __init__(self, channels=(16, 32, 64, 128), padding_mode='zero',
                 dice_smooth=1.0):
        super().__init__()
        if len(channels) != 4 or any(c <= 0 for c in channels):
            raise ValueError('Exactly four positive encoder channel counts required')
        if dice_smooth <= 0:
            raise ValueError('dice_smooth must be positive')
        self.dice_smooth = float(dice_smooth)
        self.boundary = LaplacianBoundary(padding_mode)
        self.heads = nn.ModuleList(_BFAHead(c) for c in channels)

    def forward(self, features, labeled_labels):
        if len(features) != 5:
            raise ValueError('Expected all five original encoder feature maps')
        mask = _foreground(labeled_labels)
        labeled_count = mask.shape[0]
        target = self.boundary(labeled_labels)
        ratios = target.flatten(1).mean(1)
        logits, upsampled_logits, losses, diagnostics = [], [], [], []
        for feature, head in zip(features[:4], self.heads):
            if feature.ndim != 5 or feature.shape[0] < labeled_count:
                raise ValueError('Feature batch must include all labeled samples')
            prediction = head(feature[:labeled_count])
            upsampled = prediction
            if prediction.shape[2:] != target.shape[2:]:
                upsampled = F.interpolate(prediction, size=target.shape[2:],
                    mode='trilinear', align_corners=True)
            logits.append(prediction)
            upsampled_logits.append(upsampled)
            losses.append(binary_boundary_loss(upsampled, target,
                                               epsilon=self.dice_smooth))
            diagnostics.append({
                'shape': list(prediction.shape),
                'native_shape': list(prediction.shape),
                'upsampled_shape': list(upsampled.shape),
                'target_shape': list(target.shape),
                'positive_ratio': ratios.mean().item(),
                'per_sample_positive_ratio': ratios.tolist(),
                'foreground_ratio': mask.mean().item(),
                'vanished_foreground_samples': 0,
                'empty_boundary_samples': int((ratios == 0).sum().item()),
                'high_coverage_samples': int((ratios >= 0.8).sum().item()),
            })
        return BFABoundaryResult(list(features), logits, upsampled_logits, target,
                                 [target] * 4, losses, diagnostics)


def feature_boundary_gradient(result, index):
    """Inspect dLi/dFi without detaching features or changing parameter .grad."""
    if index < 0 or index >= len(result.losses):
        raise IndexError('Boundary feature index must select one of four heads')
    return torch.autograd.grad(result.losses[index], result.features[index],
        retain_graph=True, create_graph=False, allow_unused=False)[0]
