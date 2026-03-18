"""3-D semantic segmentation trainer (slice-based)."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn

from datamint.dataset import VolumeDataset

from .lightning_modules import SegmentationModule
from .segmentation_trainer import SegmentationTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.entities import Project

class SemanticSegmentation3DTrainer(SegmentationTrainer):
    """Trainer for 3-D semantic segmentation via per-slice 2-D training.

    Builds a :class:`~datamint.dataset.VolumeDataset`, slices it along
    the chosen axis, and trains a 2-D segmentation model on individual
    slices.

    Args:
        slice_axis: Slicing axis — ``'axial'``, ``'sagittal'``,
            ``'coronal'``, or an integer axis index.
        encoder_name: SMP encoder backbone.
        in_channels: Number of input channels.

    Example::

        trainer = SemanticSegmentation3DTrainer(
            project='CT_Liver',
            slice_axis='axial',
        )
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        slice_axis: str | int = 'axial',
        encoder_name: str = 'resnet34',
        in_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.slice_axis = slice_axis
        self.encoder_name = encoder_name
        self.in_channels = in_channels

    # ── Template hooks ──────────────────────────────────────────

    def _build_default_dataset(self, project: 'str | Project'):
        vol_ds = VolumeDataset(
            project=project,
            return_as_semantic_segmentation=True,
            semantic_seg_merge_strategy='union',
            allow_external_annotations=True,
            include_unannotated=False,
        )
        return vol_ds.slice(axis=self.slice_axis)

    def _build_default_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Any],
    ) -> L.LightningModule:
        return SegmentationModule(
            arch='UnetPlusPlus',
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            num_classes=len(self.dataset.seglabel_list),
            loss_fn=loss_fn,
            metrics_factories=metrics,
        )

    def _default_train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(), # Imagenet stats is the default
            ToTensorV2(),
        ])

    def _default_eval_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ])
