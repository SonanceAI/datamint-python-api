from __future__ import annotations

from pathlib import Path
from typing import Any

import pydicom
from pydantic import field_validator

from .base_geometry import BaseGeometryAnnotation
from .geometry import CoordinateSystem, LineGeometry


class LineAnnotation(BaseGeometryAnnotation):
    """Typed line annotation entity."""

    geometry: LineGeometry | None = None

    def __init__(self, geometry: LineGeometry | dict[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs.setdefault('annotation_type', 'line')
        kwargs.setdefault('scope', 'frame' if kwargs.get('frame_index') is not None else 'image')
        super().__init__(geometry=geometry, **kwargs)

    @field_validator('geometry', mode='before')
    @classmethod
    def _validate_geometry(cls, value: Any) -> LineGeometry | None:
        return cls._coerce_geometry(value, LineGeometry)

    @classmethod
    def from_points(
        cls,
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        *,
        identifier: str,
        frame_index: int | None = None,
        dicom_metadata: pydicom.Dataset | str | Path | None = None,
        coords_system: CoordinateSystem = 'pixel',
        **kwargs: Any,
    ) -> 'LineAnnotation':
        geometry = LineGeometry.from_coordinates(
            point1,
            point2,
            coords_system=coords_system,
            frame_index=frame_index,
            dicom_metadata=dicom_metadata,
        )
        return cls(identifier=identifier, frame_index=frame_index, geometry=geometry, **kwargs)