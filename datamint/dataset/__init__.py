"""
Datamint Dataset module.

Provides specialized dataset classes for different medical imaging modalities:
- ImageDataset: 2D images (X-rays, pathology, single-frame DICOM)
- VideoDataset: Temporal sequences (videos, multi-frame DICOM)
- VolumetricDataset: 3D volumes (NIfTI, CT, MRI)

Use `build_dataset()` for automatic type detection, or instantiate directly.
"""

# New modular architecture
from .base import DatamintBaseDataset, DatamintDatasetException
from .factory import build_dataset
from .image_dataset import ImageDataset, detection_collate_fn
from .multiframe_dataset import MultiFrameDataset
from .sliced_dataset import SlicedVolumeDataset
from .sliced_video_dataset import SlicedVideoDataset
from .split_result import SplitResult
from .video_dataset import VideoDataset
from .volume_dataset import VolumeDataset

__all__ = [
    # Core
    'DatamintBaseDataset',
    'DatamintDatasetException',
    # Specialized datasets
    'ImageDataset',
    'MultiFrameDataset',
    'SlicedVideoDataset',
    'SlicedVolumeDataset',
    'SplitResult',
    'VideoDataset',
    'VolumeDataset',
    # Factory
    'build_dataset',
    'detection_collate_fn',
]