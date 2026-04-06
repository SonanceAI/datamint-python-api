"""Base segmentation module for ``segmentation_models_pytorch`` architectures."""
from __future__ import annotations

from typing import Any

from ..segmentation_module import SegmentationModule


class SMPSegmentationModule(SegmentationModule):
    """Base segmentation module for architectures from ``segmentation_models_pytorch``.

    Handles SMP-specific construction parameters shared across all SMP
    architectures.  Subclasses implement :meth:`_build_model` to return the
    concrete SMP model.

    Args:
        encoder_name: Backbone encoder (e.g. ``'resnet34'``).
        encoder_weights: Pre-trained weights to initialise the encoder with.
            Defaults to ``'imagenet'``.
        All remaining keyword arguments are forwarded to
        :class:`~datamint.lightning.trainers.lightning_modules.SegmentationModule`.
    """

    def __init__(
        self,
        encoder_name: str = 'resnet34',
        encoder_weights: str | None = 'imagenet',
        **kwargs: Any,
    ) -> None:
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        super().__init__(**kwargs)