from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

from ..resource import Resource

if TYPE_CHECKING:
    from medimgkit import ViewPlane
    import numpy as np
    from ..sliced_resource import SlicedVolumeResource


class VolumeResource(Resource):
    """Represents a volumetric resource, such as a 3D CT or MRI scan."""

    resource_kind: ClassVar[str] = 'volume'
    resource_priority: ClassVar[int] = 30
    storage_aliases: ClassVar[tuple[str, ...]] = ('VolumeResource', 'VolumeResourceHandler')

    @property
    def frame_count(self) -> int:
        return self.get_depth()

    def get_depth(self) -> int:
        frame_count = self._coerce_int(self._metadata_value('frame_count'))
        if frame_count is None:
            raise ValueError(f"Cannot determine frame count for volume resource {self.filename!r}")
        return frame_count

    def get_slice_resource(self, axis: 'ViewPlane', index: int) -> 'SlicedVolumeResource':
        return super().get_slice_resource(axis, index)

    def get_slice(self, axis: 'ViewPlane', index: int) -> 'np.ndarray':
        return super().get_slice(axis, index)

    def iter_slices(self, axis: 'ViewPlane') -> list['SlicedVolumeResource']:
        return super().iter_slices(axis)