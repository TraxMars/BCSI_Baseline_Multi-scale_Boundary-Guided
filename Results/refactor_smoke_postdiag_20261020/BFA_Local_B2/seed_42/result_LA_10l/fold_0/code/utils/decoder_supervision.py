"""Labeled-only decoder segmentation supervision, independent of the VNet.

The three inputs are skip-add features immediately before block_six,
block_seven, and block_eight. The original final prediction is optionally
included for reporting; its supervised loss belongs to the MT main loss and
is never calculated here.
"""
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from utils.losses import DiceLoss


@dataclass
class DecoderSupervisionResult:
    side_logits: List[Tensor]
    side_targets: List[Tensor]
    side_losses: List[Tensor]
    final_logits: Optional[Tensor] = None
    final_target: Optional[Tensor] = None

    @property
    def total_loss(self) -> Tensor:
        """Equal mean of three side losses; excludes the original final loss."""
        return torch.stack(self.side_losses).mean()

    @property
    def outputs(self) -> List[Tensor]:
        """Three auxiliary outputs and, when supplied, the original final."""
        return self.side_logits + ([] if self.final_logits is None else [self.final_logits])

    @property
    def targets(self) -> List[Tensor]:
        return self.side_targets + ([] if self.final_target is None else [self.final_target])


class DecoderDeepSupervision(nn.Module):
    """Three independent 1-cubed heads with the unchanged MT CE + Dice loss.

    ``labeled_labels`` determines the labeled prefix length. Each head only
    receives that prefix, preserving the original batch feature graph for
    the main MT objective. Construct this module after Student and Teacher
    under an RNG fork when exact baseline initialization is required.
    """

    def __init__(self, channels: Sequence[int] = (128, 64, 32), num_classes: int = 2):
        super().__init__()
        if len(channels) != 3 or any(channel <= 0 for channel in channels):
            raise ValueError('Decoder supervision requires three positive channel counts.')
        if num_classes < 2:
            raise ValueError('Decoder segmentation supervision requires at least two classes.')
        self.channels = tuple(channels)
        self.num_classes = num_classes
        self.heads = nn.ModuleList(nn.Conv3d(channel, num_classes, kernel_size=1)
                                   for channel in self.channels)
        self.dice_loss = DiceLoss(num_classes)

    def forward(self, decoder_features: Sequence[Tensor], labeled_labels: Tensor,
                final_logits: Optional[Tensor] = None) -> DecoderSupervisionResult:
        if len(decoder_features) != 3:
            raise ValueError('Expected three decoder features at /8, /4, and /2.')
        if labeled_labels.ndim != 4 or labeled_labels.shape[0] == 0:
            raise ValueError('Expected nonempty labeled masks with shape [N, D, H, W].')
        labeled_count = labeled_labels.shape[0]
        if labeled_labels.dtype != torch.long:
            raise ValueError('Labeled segmentation masks must have dtype torch.long.')

        side_logits, side_targets, side_losses = [], [], []
        for feature, channels, head in zip(decoder_features, self.channels, self.heads):
            if feature.ndim != 5 or feature.shape[1] != channels:
                raise ValueError('A decoder feature has an unexpected shape or channel count.')
            if feature.shape[0] < labeled_count:
                raise ValueError('Decoder feature batch is shorter than the labeled prefix.')
            logits = head(feature[:labeled_count])
            target = F.interpolate(labeled_labels.unsqueeze(1).float(),
                                   size=logits.shape[-3:], mode='nearest')[:, 0].long()
            loss = 0.5 * (F.cross_entropy(logits, target) +
                          self.dice_loss(logits, target, softmax=True))
            side_logits.append(logits)
            side_targets.append(target)
            side_losses.append(loss)

        final_labeled = None
        if final_logits is not None:
            if (final_logits.ndim != 5 or final_logits.shape[0] < labeled_count or
                    final_logits.shape[1] != self.num_classes or
                    final_logits.shape[-3:] != labeled_labels.shape[-3:]):
                raise ValueError('Final logits must match original labeled mask spatial dimensions.')
            final_labeled = final_logits[:labeled_count]
        return DecoderSupervisionResult(side_logits, side_targets, side_losses,
                                        final_labeled,
                                        labeled_labels if final_labeled is not None else None)
