"""Base segmentation module for ``segmentation_models_pytorch`` architectures."""
from __future__ import annotations

from typing import Any

from ..segmentation_module import SegmentationModule


class SMPSegmentationModule(SegmentationModule):
    """Base segmentation module for architectures from ``segmentation_models_pytorch``.

    Handles SMP-specific construction parameters shared across all SMP
    architectures.  Subclasses implement :meth:`_build_model` to return the
    concrete SMP model.

    """
    pass
