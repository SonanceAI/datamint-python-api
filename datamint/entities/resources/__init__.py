"""Specialized resource entity types."""

from .dicom_resource import DICOMResource
from .image_resource import ImageResource
from .nifti_resource import NiftiResource
from .video_resource import VideoResource
from .volume_resource import VolumeResource

__all__ = [
    'DICOMResource',
    'ImageResource',
    'NiftiResource',
    'VideoResource',
    'VolumeResource',
]