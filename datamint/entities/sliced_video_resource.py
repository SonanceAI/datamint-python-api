"""
SlicedVideoResource - Proxy for a single frame of a video Resource.

Analogous to :class:`SlicedVolumeResource` but simplified for temporal
data — videos always slice along the frame (temporal) axis.
"""
from __future__ import annotations
import gzip
import logging
from typing import Any, TYPE_CHECKING
from functools import cached_property

from medimgkit.readers import read_array_normalized
from datamint.entities.cache_manager import CacheManager
import numpy as np

if TYPE_CHECKING:
    from datamint.entities import Resource

_LOGGER = logging.getLogger(__name__)

# Cache key for parsed frame numpy arrays
_FRAME_ARRAY_CACHEKEY = "frame_array"


class SlicedVideoResource:
    """Proxy that presents a single frame of a video Resource.

    Wraps a :class:`Resource` and represents a specific frame by index.
    Uses gzip-compressed ``.npy.gz`` files on disk for caching, with an
    in-memory LRU cache managed by :class:`CacheManager`.

    Args:
        parent: The original video Resource.
        frame_index: The index of the frame in the video.
        frame_cache: Shared :class:`CacheManager` for disk-based frame caching.
    """

    _CACHE_MANAGER_NAMESPACE = "sliced_video_frames"

    def __init__(
        self,
        parent: Resource,
        frame_index: int,
        frame_cache: CacheManager | None = None,
    ):
        self._parent = parent
        self.frame_index = frame_index
        if frame_cache is None:
            frame_cache = CacheManager(SlicedVideoResource._CACHE_MANAGER_NAMESPACE)
        self._frame_cache = frame_cache

    @staticmethod
    def slice_over(
        resource: Resource,
        frame_cache: CacheManager | None = None,
    ) -> list[SlicedVideoResource]:
        """Expand a video resource into per-frame proxy resources.

        Args:
            resource: The video Resource to expand.
            frame_cache: Shared cache for decoded frames.

        Returns:
            List of :class:`SlicedVideoResource`, one per frame.
        """
        num_frames = resource.get_depth()
        return [
            SlicedVideoResource(resource, i, frame_cache)
            for i in range(num_frames)
        ]

    def get_depth(self) -> int:
        """A single frame has depth 1."""
        return 1

    def _get_version_info(self) -> dict:
        """Get version info from the parent resource for cache validation."""
        return {
            'created_at': self._parent.created_at,
            'deleted_at': self._parent.deleted_at,
            'size': self._parent.size,
        }

    def _frame_cache_entity_id(self) -> str:
        return f"{self._parent.id}:frame{self.frame_index}"

    def fetch_frame_data(self) -> np.ndarray:
        """Fetch the frame as a ``(C, H, W)`` array.

        Returns:
            Frame array with shape ``(C, H, W)``.
        """
        version_info = self._get_version_info()
        cache_entity_id = self._frame_cache_entity_id()

        cached_frame = self._frame_cache.get(
            cache_entity_id,
            _FRAME_ARRAY_CACHEKEY,
            version_info,
        )
        if cached_frame is not None:
            return np.ascontiguousarray(cached_frame)

        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        frame, self.data_metainfo = read_array_normalized(raw, return_metainfo=True, 
                                                        index=self.frame_index)  # frame.shape is (C, H, W)
        _LOGGER.debug(f'Fetched raw frame data for frame index {self.frame_index}. Frame shape: {frame.shape}')
        frame = np.ascontiguousarray(frame)

        gz_path = self._frame_cache.get_expected_path(cache_entity_id, _FRAME_ARRAY_CACHEKEY)
        gz_path = gz_path.with_suffix('.npy.gz')
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(gz_path), 'wb', compresslevel=4) as f:
            np.save(f, frame)

        self._frame_cache.register_file_location(
            cache_entity_id,
            _FRAME_ARRAY_CACHEKEY,
            file_path=gz_path,
            version_info=version_info,
            mimetype='application/gzip',
            data=frame,
        )

        _LOGGER.debug(f'Frame shape: {frame.shape}')

        return frame

    @cached_property
    def data_metainfo(self) -> dict:
        """Video metadata. Loaded once and cached for the lifetime of this resource."""
        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        _, metainfo = read_array_normalized(raw, return_metainfo=True)
        return metainfo

    @property
    def parent_resource(self) -> Resource:
        """The original video Resource being proxied."""
        return self._parent

    def __repr__(self) -> str:
        return (
            f"SlicedVideoResource(filename='{self._parent.filename}', "
            f"frame={self.frame_index})"
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            parent = super().__getattribute__('_parent')
            return getattr(parent, name)
