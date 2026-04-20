from __future__ import annotations

from pathlib import Path
from typing import Any

import pydicom
from pydantic import field_validator
from nibabel.nifti1 import Nifti1Image

from .base_geometry import BaseGeometryAnnotation
from .geometry import BoxGeometry, CoordinateSystem


class BoxAnnotation(BaseGeometryAnnotation):
    """Typed box annotation entity."""

    geometry: BoxGeometry | None = None

    def __init__(self, geometry: BoxGeometry | dict[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs.setdefault('annotation_type', 'square')
        kwargs.setdefault('scope', 'frame' if kwargs.get('frame_index') is not None else 'image')
        super().__init__(geometry=geometry, **kwargs)

    @field_validator('geometry', mode='before')
    @classmethod
    def _validate_geometry(cls, value: Any) -> BoxGeometry | None:
        return cls._coerce_geometry(value, BoxGeometry)

    @classmethod
    def from_points(
        cls,
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        *,
        identifier: str,
        frame_index: int | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
        coords_system: CoordinateSystem = 'pixel',
        **kwargs: Any,
    ) -> 'BoxAnnotation':
        geometry = BoxGeometry.from_coordinates(
            point1,
            point2,
            coords_system=coords_system,
            frame_index=frame_index,
            metadata=metadata,
        )
        return cls(identifier=identifier, frame_index=frame_index, geometry=geometry, **kwargs)