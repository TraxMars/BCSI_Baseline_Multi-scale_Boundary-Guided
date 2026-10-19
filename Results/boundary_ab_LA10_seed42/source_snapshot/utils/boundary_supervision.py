"""Student-only multi-scale boundary heads and explicitly routed gradients.

No feature is detached: ``result.features`` and ``result.losses`` remain usable
with autograd.grad(losses[i], features[i]) before the training backward call.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


STAGE_MODULES = (
    ('block_one',),
    ('block_one_dw', 'block_two'),
    ('block_two_dw', 'block_three'),
    ('block_three_dw', 'block_four'),
    ('block_four_dw', 'block_five'),
)


def encoder_stage_parameters(encoder):
    """Partition every encoder parameter exactly once, including downsamplers."""
    groups = tuple(tuple(p for name in names
                         for p in getattr(encoder, name).parameters())
                   for names in STAGE_MODULES)
    ids = [id(p) for group in groups for p in group]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate parameters in encoder stage groups')
    if set(ids) != {id(p) for p in encoder.parameters()}:
        raise ValueError('Encoder stage groups omit or add parameters')
    return groups


@torch.no_grad()
def boundary_target(labels, size):
    """Nearest resize, then a 3x3x3 dilation minus erosion on foreground.

    Replicated padding treats the volume edge as continuing the nearest label;
    it does not invent a boundary along a cropped all-foreground volume edge.
    Accept [N,D,H,W] or [N,1,D,H,W], with foreground defined by label > 0.
    """
    if labels.ndim == 4:
        labels = labels.unsqueeze(1)
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError('Expected labels [N,D,H,W] or [N,1,D,H,W]')
    mask = F.interpolate((labels > 0).float(), size=tuple(size), mode='nearest')
    padded = F.pad(mask, (1, 1, 1, 1, 1, 1), mode='replicate')
    dilation = F.max_pool3d(padded, 3, stride=1)
    erosion = -F.max_pool3d(-padded, 3, stride=1)
    return dilation - erosion, mask


def binary_boundary_loss(logits, target, epsilon=1e-5):
    """Equal BCEWithLogits and per-volume spatial soft Dice, no double sigmoid."""
    if logits.shape != target.shape or logits.ndim != 5 or logits.shape[1] != 1:
        raise ValueError('Boundary logits/targets must match [N,1,D,H,W]')
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    dims = (2, 3, 4)
    dice = (2 * (probability * target).sum(dims) + epsilon) / (
        probability.sum(dims) + target.sum(dims) + epsilon)
    return 0.5 * bce + 0.5 * (1 - dice).mean()


@dataclass
class BoundaryResult:
    features: list
    logits: list
    targets: list
    losses: list
    diagnostics: list

    @property
    def total_loss(self):
        return torch.stack(self.losses).mean()


class MultiScaleBoundarySupervision(nn.Module):
    def __init__(self, channels=(16, 32, 64, 128, 256)):
        super().__init__()
        if len(channels) != 5:
            raise ValueError('Exactly five encoder scales are required')
        self.heads = nn.ModuleList(nn.Conv3d(c, 1, 1) for c in channels)

    def forward(self, features, labeled_labels):
        if len(features) != len(self.heads) or len(labeled_labels) == 0:
            raise ValueError('Expected five feature maps and nonempty labeled batch')
        logits, targets, losses, diagnostics = [], [], [], []
        source_nonempty = (labeled_labels > 0).flatten(1).any(1)
        for feature, head in zip(features, self.heads):
            prediction = head(feature[:len(labeled_labels)])
            target, mask = boundary_target(labeled_labels, prediction.shape[2:])
            ratios = target.flatten(1).mean(1)
            if not torch.all((target == 0) | (target == 1)):
                raise AssertionError('Boundary target must be binary and finite')
            vanished = source_nonempty & ~mask.flatten(1).bool().any(1)
            diagnostics.append({
                'shape': list(prediction.shape),
                'positive_ratio': ratios.mean().item(),
                'per_sample_positive_ratio': ratios.tolist(),
                'foreground_ratio': mask.mean().item(),
                'vanished_foreground_samples': int(vanished.sum().item()),
                'empty_boundary_samples': int((ratios == 0).sum().item()),
                'high_coverage_samples': int((ratios >= 0.8).sum().item()),
            })
            logits.append(prediction)
            targets.append(target)
            losses.append(binary_boundary_loss(prediction, target))
        return BoundaryResult(list(features), logits, targets, losses, diagnostics)


def local_boundary_gradients(result, stage_parameters, heads, boundary_weight,
                             stage_indices=None):
    """Get only each loss's OWN stage/head gradients, weighted by weight / 5.

    All calls retain the shared graph for subsequent MT backward. allow_unused
    is deliberately False: a missing gradient indicates a broken stage mapping.
    autograd.grad never populates .grad on upstream or downstream parameters.
    stage_indices isolates losses for the propagation audit.
    """
    indices = range(len(result.losses)) if stage_indices is None else stage_indices
    gradients = []
    for index in indices:
        parameters = tuple(stage_parameters[index]) + tuple(heads[index].parameters())
        values = torch.autograd.grad(
            result.losses[index] * (boundary_weight / len(result.losses)),
            parameters, retain_graph=True, create_graph=False, allow_unused=False)
        gradients.extend((p, g.detach()) for p, g in zip(parameters, values))
    return gradients


def add_local_gradients(gradients):
    for parameter, gradient in gradients:
        if parameter.grad is None:
            parameter.grad = gradient
        else:
            parameter.grad.add_(gradient)
