"""
VolumeDataset - Dataset for 3D medical volumes.

Handles NIfTI volumes, DICOM series, and other 3D medical imaging data
with support for different slice orientations and affine preservation.
"""
import logging
from typing import TYPE_CHECKING

from .multiframe_dataset import MultiFrameDataset

if TYPE_CHECKING:
    from .sliced_dataset import SlicedVolumeDataset

_LOGGER = logging.getLogger(__name__)


# Axis mapping for anatomical orientations
SLICE_AXIS_MAP = {
    'axial': 0,      # slicing along depth (superior-inferior)
    'coronal': 1,    # slicing along height (anterior-posterior)
    'sagittal': 2,   # slicing along width (left-right)
}


class VolumeDataset(MultiFrameDataset):
    """Dataset for 3D medical volumes.

    Handles NIfTI (3D/4D), DICOM series, and other volumetric data.
    Inherits multi-frame loading and augmentation from :class:`MultiFrameDataset`.
    """

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"VolumeDataset\n{base}"

    def slice(self, axis: str | int = 'axial') -> 'SlicedVolumeDataset':
        """Create a 2D dataset by slicing this volume along an axis.

        Each 3D volume is expanded into multiple 2D slices, one per depth index
        along the given axis. The returned dataset yields 2D items with shape
        ``(C, H, W)`` instead of ``(C, D, H, W)``.

        Parsed volumes are cached to disk as gzip-compressed ``.npy.gz`` files.
        A shared in-memory LRU cache also keeps recently used full volumes to
        avoid repeated decompression when iterating neighboring slices.

        Args:
            axis: Slice orientation. One of ``'axial'`` (depth), ``'coronal'``
                (height), ``'sagittal'`` (width), or an integer axis index (0--2).

        Returns:
            A :class:`SlicedVolumeDataset` that iterates over individual 2D slices.

        Example::

            vol_ds = VolumeDataset(project='my_ct_project')
            sliced = vol_ds.slice(axis='axial')
            print(len(sliced))  # total number of axial slices across all volumes
            item = sliced[0]
            print(item['image'].shape)  # (C, H, W)
        """
        from .sliced_dataset import SlicedVolumeDataset

        return SlicedVolumeDataset.from_dataset(
            parent_dataset=self,
            slice_axis=axis,
        )
