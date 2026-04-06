"""UNet++ segmentation module."""
from __future__ import annotations

from torch import nn

from .smp_module import SMPSegmentationModule


class UNetPPModule(SMPSegmentationModule):
    """Segmentation module using the UNet++ architecture from ``segmentation_models_pytorch``.

    Args:
        All arguments are forwarded to
        :class:`~datamint.lightning.trainers.lightning_modules.segmentation_modules.SMPSegmentationModule`.
    """

    def _build_model(self) -> nn.Module:
        import segmentation_models_pytorch as smp

        return smp.UnetPlusPlus(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
        )
