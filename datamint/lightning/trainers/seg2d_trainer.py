"""2-D semantic segmentation trainer."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING
from collections.abc import Callable

import lightning as L
from torch import nn

from datamint.dataset import ImageDataset

from .lightning_modules import UNetPPModule
from .segmentation_trainer import SegmentationTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.entities import Project

class SemanticSegmentation2DTrainer(SegmentationTrainer):
    """Trainer for 2-D semantic segmentation.

    Default model: **UNet++** (``segmentation_models_pytorch``) with a
    ``resnet34`` encoder pretrained on ImageNet.

    Args:
        in_channels: Number of input image channels.  Defaults to ``3``.
        All remaining keyword arguments are forwarded to
        :class:`~datamint.lightning.trainers.base_trainer.BaseTrainer`.

    Example::

        trainer = SemanticSegmentation2DTrainer(project='BUS_Segmentation')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        in_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels

    # ── Template hooks ──────────────────────────────────────────

    def _build_dataset(self, project: 'str | Project') -> ImageDataset:
        return ImageDataset(
            project=project,
            return_as_semantic_segmentation=True,
            semantic_seg_merge_strategy='union',
            allow_external_annotations=True,
            include_unannotated=False,
        )

    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(), # Imagenet stats is the default
            ToTensorV2(),
        ])

    def _eval_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ])


class UNetPPTrainer(SemanticSegmentation2DTrainer):
    """Convenience trainer pre-configured for UNet++ with stronger augmentations.

    Adds elastic transform and grid distortion to the default training
    pipeline — augmentations that are particularly effective for medical
    image segmentation.

    Example::

        trainer = UNetPPTrainer(
            project='BUS_Segmentation',
            encoder_name='resnet34',)
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        encoder_name: str = 'resnet34',
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_name = encoder_name

    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _build_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Callable],
    ) -> L.LightningModule:
        return UNetPPModule(
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            num_classes=len(self.dataset.seglabel_list),
            loss_fn=loss_fn,
            metrics_factories=metrics,
            class_names=list(self.dataset.seglabel_list),
            image_size=self.image_size,
        )