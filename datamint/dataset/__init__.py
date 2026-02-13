"""
Datamint Dataset module.

Provides specialized dataset classes for different medical imaging modalities:
- ImageDataset: 2D images (X-rays, pathology, single-frame DICOM)
- VideoDataset: Temporal sequences (videos, multi-frame DICOM)
- VolumetricDataset: 3D volumes (NIfTI, CT, MRI)

Use `create_dataset()` for automatic type detection, or instantiate directly.
"""

# New modular architecture
from .base import DatamintBaseDataset, DatamintDatasetException
from .image_dataset import ImageDataset
from .volume_dataset import VolumeDataset
from .sliced_dataset import SlicedVolumeDataset, SlicedVolumeResource

__all__ = [
    # Core
    'DatamintBaseDataset',
    'DatamintDatasetException',
    # Specialized datasets
    'ImageDataset',
    'VolumeDataset',
    'SlicedVolumeDataset',
    'SlicedVolumeResource',
]