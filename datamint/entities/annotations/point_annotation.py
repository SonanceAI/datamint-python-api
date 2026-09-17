from __future__ import annotations

from typing import Any

import pydicom
from medimgkit import ViewPlane
from nibabel.nifti1 import Nifti1Image
from pydantic import field_validator

from .base_geometry import BaseGeometryAnnotation
from .geometry import CoordinateSystem, PointGeometry


class PointAnnotation(BaseGeometryAnnotation):
    """Typed point annotation entity."""

    geometry: PointGeometry | None = None

    def __init__(self, geometry: PointGeometry | dict[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs.setdefault('annotation_type', 'point')
        kwargs.setdefault('scope', 'frame' if kwargs.get('frame_index') is not None else 'image')
        super().__init__(geometry=geometry, **kwargs)

    @field_validator('geometry', mode='before')
    @classmethod
    def _validate_geometry(cls, value: Any) -> PointGeometry | None:
        return cls._coerce_geometry(value, PointGeometry)

    @classmethod
    def from_point(
        cls,
        point: tuple[int, int] | tuple[float, float, float],
        *,
        identifier: str,
        frame_index: int | None = None,
        slice_plane: ViewPlane | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
        coords_system: CoordinateSystem = 'pixel',
        **kwargs: Any,
    ) -> PointAnnotation:
        geometry = PointGeometry.from_coordinates(
            point,
            coords_system=coords_system,
            frame_index=frame_index,
            slice_plane=slice_plane,
            metadata=metadata,
        )
        return cls(identifier=identifier, frame_index=frame_index, geometry=geometry, **kwargs)
