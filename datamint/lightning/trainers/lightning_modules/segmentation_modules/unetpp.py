"""UNet++ segmentation module."""
from __future__ import annotations
from typing import Any
from typing_extensions import override
from collections.abc import Callable

from torch import Tensor, nn

import albumentations as A
import segmentation_models_pytorch as smp

from .smp_module import SMPSegmentationModule


class UNetPPModule(SMPSegmentationModule):
    """Segmentation module using the UNet++ architecture from ``segmentation_models_pytorch``.

    Args:
        All arguments are forwarded to
        :class:`~datamint.lightning.trainers.lightning_modules.segmentation_modules.SMPSegmentationModule`.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss_fn: nn.Module | None = None,
        metrics_factories: dict[str, Callable[[], Any]] = {},
        class_names: list[str] | None = None,
        image_size: tuple[int, int] | None = None,
        lr: float = 1e-4,
        encoder_name: str = 'resnet34',
        encoder_weights: str | None = 'imagenet',
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights

        super().__init__(transform=transform,
                         loss_fn=loss_fn,
                         metrics_factories=metrics_factories,
                         class_names=class_names,
                         lr=lr)

        self.model = smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)