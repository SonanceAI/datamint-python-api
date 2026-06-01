"""DeepLab v3+ segmentation module."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any

import albumentations as A
from torch import Tensor, nn
from typing_extensions import override

import segmentation_models_pytorch as smp

from .smp_module import SMPSegmentationModule


class DeepLabV3PlusModule(SMPSegmentationModule):
    """Segmentation module using the DeepLab v3+ architecture from ``segmentation_models_pytorch``.

    Args:
        in_channels: Number of input image channels.
        num_classes: Number of segmentation classes excluding background.
        loss_fn: Loss module.
        metrics_factories: ``{name: callable}`` where each callable returns a fresh metric.
        class_names: Human-readable label for each class.
        image_size: ``(height, width)`` used during inference.
        lr: Learning rate for AdamW.
        encoder_name: SMP encoder backbone (e.g. ``'resnet34'``, ``'efficientnet-b4'``).
        encoder_weights: Pre-trained weights to load. ``'imagenet'`` by default.
        decoder_atrous_rates: Dilation rates for the ASPP module.
            Controls the multi-scale receptive field that is DeepLab v3+'s
            core architectural feature. SMP default is ``(12, 24, 36)``.
        transform: Albumentations transform applied during inference.
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
        decoder_atrous_rates: tuple[int, int, int] = (12, 24, 36),
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.decoder_atrous_rates = decoder_atrous_rates

        super().__init__(
            transform=transform,
            loss_fn=loss_fn,
            metrics_factories=metrics_factories,
            class_names=class_names,
            lr=lr,
        )

        self.model = smp.DeepLabV3Plus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
            decoder_atrous_rates=self.decoder_atrous_rates,
        )

    @override
    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)
