"""True 3-D volume segmentation trainer (no slicing)."""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

import albumentations as A
from albumentations.pytorch import ToTensorV2

from datamint.dataset import VolumeDataset
from datamint.lightning.datamodule import DatamintDataModule

from .segmentation_trainer import SegmentationTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project


class _BCEDiceLoss3D(nn.Module):
    """Combined BCE + soft Dice for 3-D multi-label segmentation.

    Identical logic to ``_BCEDiceLoss`` in segmentation_trainer.py but sums
    over the 3 spatial axes (dims 0, 2, 3, 4) instead of 2 (0, 2, 3).

    Expects:
        pred:   ``(B, C, D, H, W)`` logits
        target: ``(B, C, D, H, W)`` binary masks (float)
    """

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, reduction: str = 'mean'
    ) -> torch.Tensor:
        target = target.float()
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction=reduction)
        probs = torch.sigmoid(pred)
        dims = (0, 2, 3, 4)                         # batch + 3 spatial axes
        intersection = (probs * target).sum(dim=dims)
        cardinality   = probs.sum(dim=dims) + target.sum(dim=dims)
        dice_per_class = (2.0 * intersection + 1e-6) / (cardinality + 1e-6)
        return bce + (1.0 - dice_per_class.mean())


class VolumeSegmentationTrainer(SegmentationTrainer):
    """Abstract trainer for true 3-D volumetric segmentation.

    Uses :class:`~datamint.dataset.VolumeDataset` directly (no slicing).
    Subclasses must implement :meth:`_build_model`.

    Training uses the full volume (or a random patch cropped inside the
    Lightning module's ``training_step``).  Eval uses one full volume at a
    time (``eval_batch_size=1``); sliding-window inference is handled by the
    Lightning module.

    Args:
        patch_crop_size: ``(D, H, W)`` patch size used during training.
            Forwarded to the Lightning module for random cropping.
        batch_size: Number of volumes per training batch.  Defaults to ``1``
            because 3-D volumes cannot be stacked unless they are all the
            same spatial size — use ``1`` unless you are certain all your
            volumes match.
        All remaining keyword arguments are forwarded to
        :class:`~datamint.lightning.trainers.base_trainer.BaseTrainer`.

    Example::

        class MyTrainer(VolumeSegmentationTrainer):
            def _build_model(self, loss_fn, metrics):
                return MyLightningModule(...)

        trainer = MyTrainer(project='CT_Liver')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        patch_crop_size: tuple[int, int, int] = (128, 128, 128),
        batch_size: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(batch_size=batch_size, **kwargs)
        self.patch_crop_size = patch_crop_size

    # ── Template hooks ───────────────────────────────────────────

    def _build_dataset(self, project: 'str | Project', **kwargs: Any) -> VolumeDataset:
        default_params: dict[str, Any] = dict(
            return_as_semantic_segmentation=True,
            semantic_seg_merge_strategy='union',
            allow_external_annotations=True,
            include_unannotated=False,
        )
        return VolumeDataset(project=project, **{**default_params, **kwargs})

    def _loss(self) -> nn.Module:
        return _BCEDiceLoss3D()

    def _metrics(self) -> dict[str, Callable]:
        from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

        num_classes = len(self.dataset.seglabel_list)
        return {
            'iou':  partial(MeanIoU,              num_classes=num_classes, input_format='one-hot'),
            'dice': partial(GeneralizedDiceScore,  num_classes=num_classes, input_format='one-hot'),
        }

    def _train_transform(self) -> 'BaseCompose':
        """Intensity-only transforms (no crop).

        Spatial crop is done inside the Lightning module's ``training_step``
        so that image and mask are cropped identically without needing a
        custom albumentations 3-D crop transform.
        """
        return A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ])

    def _eval_transform(self) -> 'BaseCompose':
        """Normalise only — no crop.

        Full volumes are passed to the Lightning module's sliding-window
        inference during validation and test.
        """
        return A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ])

    def _build_datamodule(
        self,
        dataset: 'DatamintBaseDataset',
        train_transform: 'BaseCompose',
        eval_transform: 'BaseCompose',
    ) -> DatamintDataModule:
        """Use ``eval_batch_size=1`` for val/test.

        Full volumes vary in spatial size across a dataset, so batching
        them requires either padding or batch_size=1.  During eval we always
        use 1 to avoid shape mismatches; the Lightning module handles
        inference via sliding window.
        """
        return DatamintDataModule(
            dataset,
            batch_size=self.batch_size,
            val_batch_size=1,
            test_batch_size=1,
            num_workers=self.num_workers,
            split_as_of_timestamp=self.split_as_of_timestamp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            pin_memory=False,
        )
