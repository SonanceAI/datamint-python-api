"""2-D semantic segmentation trainer."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

from datamint.dataset import ImageDataset

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
        # TODO: automatically check if project is composed of 3D volumes or 2D images and choose SlicedVolumeDataset vs ImageDataset accordingly.
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
            A.Normalize(),  # Imagenet stats is the default
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