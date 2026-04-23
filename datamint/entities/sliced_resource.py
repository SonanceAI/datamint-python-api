from __future__ import annotations
import gzip
import logging
from typing import Any, Literal, TYPE_CHECKING
from functools import cached_property
from medimgkit.readers import read_array_normalized
from medimgkit import dicom_utils, nifti_utils, ViewPlane
from datamint.entities.cache_manager import CacheManager
import numpy as np

if TYPE_CHECKING:
    from datamint.entities import Resource

_LOGGER = logging.getLogger(__name__)

# Cache key for parsed slice numpy arrays
_SLICE_ARRAY_CACHEKEY = "slice_array"


class SlicedVolumeResource:
    """Proxy that presents a single 2D slice of a 3D volume Resource.

    This class wraps a :class:`Resource` and represents a specific 2D slice
    along a given axis. It uses gzip-compressed ``.npy.gz`` files on disk
    for efficient storage, with an in-memory LRU cache managed by
    :class:`CacheManager`.

    The CacheManager memory cache is disabled by default globally, but the
    sliced-volume cache manager enables it by default.

    This shared cache avoids repeated gzip decompression for already-cached
    slices. Full-volume caching is intentionally not handled here.

    Args:
        parent: The original 3D volume Resource.
        slice_index: The index of the slice along the given axis.
        slice_axis: The spatial axis to slice along ('axial', 'coronal', 'sagittal').
        sliced_vols_cache: Shared :class:`CacheManager` for disk-based volume caching.
    """

    _CACHE_MANAGER_NAMESPACE = "sliced_volumes"

    def __init__(
        self,
        parent: Resource,
        slice_index: int,
        slice_axis: ViewPlane,
        sliced_vols_cache: CacheManager | None = None,
    ):
        self._parent = parent
        self.slice_index = slice_index
        self.slice_axis = slice_axis
        if sliced_vols_cache is None:
            sliced_vols_cache = CacheManager(SlicedVolumeResource._CACHE_MANAGER_NAMESPACE)
        self._volume_cache = sliced_vols_cache

    @staticmethod
    def slice_over(resource: Resource,
                   slice_axis: ViewPlane,
                   volume_cache: CacheManager | None = None) -> list[SlicedVolumeResource]:
        sliced_resources = []
        res_data = resource.fetch_file_data(auto_convert=True, use_cache=True)
        if resource.is_dicom():
            axis_size = dicom_utils.get_dim_size(res_data, slice_axis)
        elif resource.is_nifti():
            axis_size = nifti_utils.get_dim_size(res_data, slice_axis)
        else:
            raise ValueError(f"Unsupported resource type for slicing axis: {resource.filename}|{resource.mimetype}")

        # anns = resource_annotations[i]

        for s in range(axis_size):
            sliced_resources.append(
                SlicedVolumeResource(resource, s, slice_axis, volume_cache)
            )

        return sliced_resources

    def __getattr__(self, name: str) -> Any:
        """Delegate all unresolved attributes to the parent Resource."""
        if name == '_api':
            return None
        if name == '_parent':
            return super().__getattribute__('_parent')
        _properties = {'slice_axis_idx', 'slice_axis_idx_std', 'data_metainfo'}
        if name in _properties:
            return super().__getattribute__(name)
        return getattr(self._parent, name)

    def get_depth(self) -> int:
        """A single slice has depth 1."""
        return 1

    def _get_version_info(self) -> dict:
        """Get version info from the parent resource for cache validation."""
        return {
            'created_at': self._parent.created_at,
            'deleted_at': self._parent.deleted_at,
            'size': self._parent.size,
        }

    def _slice_cache_entity_id(self) -> str:
        return f"{self._parent.id}:axis{self.slice_axis}:slice{self.slice_index}"

    def fetch_slice_data(self) -> np.ndarray:
        """Fetch the 2D slice as a (C, H, W) array.

        Returns:
            Slice array with shape (C, H, W).
        """
        version_info = self._get_version_info()
        cache_entity_id = self._slice_cache_entity_id()

        cached_slice = self._volume_cache.get(
            cache_entity_id,
            _SLICE_ARRAY_CACHEKEY,
            version_info,
        )
        if cached_slice is not None:
            return np.ascontiguousarray(cached_slice)

        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        vol, self.data_metainfo = read_array_normalized(raw, return_metainfo=True)
        # vol.shape is (D,C,H,W)

        _LOGGER.debug("Slicing %s along axis %s (%s) at index %s with shape %s",
                      self._parent.filename, self.slice_axis, self.slice_axis_idx_std, self.slice_index, vol.shape
                      )
        sliced = np.take(vol, self.slice_index, axis=self.slice_axis_idx_std)
        _LOGGER.debug('Sliced shape before channel axis adjustment: %s', sliced.shape)
        # vol is (D, C, H, W); after np.take the sliced axis is removed.
        # C was at index 1. If we sliced axis 0 (D), C shifts to index 0 — already correct.
        # If we sliced axis 2 (H) or 3 (W), C stays at index 1 → move it to 0.
        channel_axis_in_result = 1 if self.slice_axis_idx_std > 1 else 0
        if channel_axis_in_result != 0:
            sliced = np.moveaxis(sliced, channel_axis_in_result, 0)
        # sliced is now (C, DIM1, DIM2)
        sliced = np.ascontiguousarray(sliced)

        _LOGGER.debug('Sliced shape before caching: %s', sliced.shape)

        gz_path = self._volume_cache.get_expected_path(cache_entity_id, _SLICE_ARRAY_CACHEKEY)
        gz_path = gz_path.with_suffix('.npy.gz')
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(gz_path), 'wb', compresslevel=4) as f:
            np.save(f, sliced)

        self._volume_cache.register_file_location(
            cache_entity_id,
            _SLICE_ARRAY_CACHEKEY,
            file_path=gz_path,
            version_info=version_info,
            mimetype='application/gzip',
            data=sliced,
        )

        _LOGGER.debug(f'Sliced shape: {sliced.shape}')

        return sliced

    @cached_property
    def data_metainfo(self) -> dict:
        """Volume metadata. Loaded once and cached for the lifetime of this resource."""
        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        _, metainfo = read_array_normalized(raw, return_metainfo=True)
        return metainfo

    @cached_property
    def slice_axis_idx(self) -> int:
        """Raw axis index for the slice axis. Computed once and cached."""
        if self._parent.is_dicom():
            return dicom_utils.get_plane_axis(self.data_metainfo, self.slice_axis)
        elif self._parent.is_nifti():
            return nifti_utils.get_plane_axis(self.data_metainfo, self.slice_axis)
        else:
            raise ValueError(
                "Unsupported resource type for slicing axis:"
                f" {self._parent.filename}|{self._parent.mimetype}|{self.slice_axis}")

    @cached_property
    def slice_axis_idx_std(self) -> int:
        """Standardised axis index for the slice axis. Computed once and cached."""
        if self._parent.is_dicom():
            return dicom_utils.rawplaneaxis2stdplaneaxis_idx(self.slice_axis_idx)
        elif self._parent.is_nifti():
            return nifti_utils.rawplaneaxis2stdplaneaxis_idx(self.slice_axis_idx)
        else:
            raise ValueError(
                f"Unsupported resource type for slicing axis: {self._parent.filename}|{self._parent.mimetype}")

    @property
    def parent_resource(self) -> Resource:
        """The original volume Resource being proxied."""
        return self._parent

    def __repr__(self) -> str:
        axis_names = {0: 'axial', 1: 'coronal', 2: 'sagittal'}
        axis_name = axis_names.get(self.slice_axis, str(self.slice_axis))
        return (
            f"SlicedVolumeResource(filename='{self._parent.filename}', "
            f"axis='{axis_name}', slice={self.slice_index})"
        )

    def is_cached(self) -> bool:
        return self._parent.is_cached()

    def __getstate__(self):
        state = super().__getstate__()
        if '_api' in state:
            _LOGGER.info("Removing _api from SlicedVolumeResource state for pickling."
                         " It shouldn't be there.")
            del state['_api']
        if 'data_metainfo' in state:
            _ = self.slice_axis_idx # ensure other cached properties are computed before deleting data_metainfo
            del state['data_metainfo']
        return state
