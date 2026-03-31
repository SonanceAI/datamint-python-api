"""Shared base for segmentation trainers (2-D and 3-D)."""
from __future__ import annotations

from collections.abc import Callable

from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from .base_trainer import BaseTrainer


class _BCEDiceLoss(nn.Module):
    """Combined binary-cross-entropy-with-logits + soft Dice loss.

    Operates on multi-label masks where each class channel is independent.

    Expects:
        pred:   ``(B, C, H, W)`` logits
        target: ``(B, C, H, W)`` binary masks (float)
    """

    def forward(self, pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
        probs = torch.sigmoid(pred)
        dims = (0, 2, 3)
        intersection = (probs * target).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + target.sum(dim=dims)
        dice_per_class = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return bce + (1.0 - dice_per_class.mean())


class SegmentationTrainer(BaseTrainer):
    """Abstract trainer for segmentation tasks.

    Provides shared defaults:

    * **Loss** – combined BCE + Dice (:class:`_BCEDiceLoss`).
    * **Metrics** – Mean IoU and Generalised Dice Score (torchmetrics).
    * **Monitor** – ``val/iou`` (maximise).
    """

    def _loss(self) -> nn.Module:
        return _BCEDiceLoss()

    def _metrics(self) -> dict[str, Callable]:
        from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

        num_classes = len(self.dataset.seglabel_list)
        return {
            'iou': partial(MeanIoU, num_classes=num_classes, input_format='one-hot'),
            'dice': partial(GeneralizedDiceScore, num_classes=num_classes, input_format='one-hot'),
        }

    def _monitor_metric(self) -> tuple[str, str]:
        return 'val/iou', 'max'
