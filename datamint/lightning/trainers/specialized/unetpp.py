from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn
from ..lightning_modules import UNetPPModule
from ..seg2d_trainer import SemanticSegmentation2DTrainer


if TYPE_CHECKING:
    from albumentations import BaseCompose


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
