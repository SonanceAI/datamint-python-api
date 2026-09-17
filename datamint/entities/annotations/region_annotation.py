from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pydicom
from medimgkit import ViewPlane
from nibabel.nifti1 import Nifti1Image
from pydantic import field_validator

from .base_geometry import BaseGeometryAnnotation
from .geometry import CoordinateSystem, RegionGeometry


class RegionAnnotation(BaseGeometryAnnotation):
    """Typed region annotation entity: an open polyline or a closed contour."""

    geometry: RegionGeometry | None = None

    def __init__(self, geometry: RegionGeometry | dict[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs.setdefault('annotation_type', 'region')
        kwargs.setdefault('scope', 'frame' if kwargs.get('frame_index') is not None else 'image')
        super().__init__(geometry=geometry, **kwargs)

    @field_validator('geometry', mode='before')
    @classmethod
    def _validate_geometry(cls, value: Any) -> RegionGeometry | None:
        return cls._coerce_geometry(value, RegionGeometry)

    @classmethod
    def from_points(
        cls,
        points: Sequence[tuple[int, int] | tuple[float, float, float]],
        *,
        identifier: str,
        closed: bool = False,
        frame_index: int | None = None,
        slice_plane: ViewPlane | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
        coords_system: CoordinateSystem = 'pixel',
        **kwargs: Any,
    ) -> RegionAnnotation:
        geometry = RegionGeometry.from_coordinates(
            points,
            closed=closed,
            coords_system=coords_system,
            frame_index=frame_index,
            slice_plane=slice_plane,
            metadata=metadata,
        )
        return cls(identifier=identifier, frame_index=frame_index, geometry=geometry, **kwargs)
