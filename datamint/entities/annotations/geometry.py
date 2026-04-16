from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
import pydicom
from medimgkit.dicom_utils import pixel_to_patient
from pydantic import BaseModel, ConfigDict, field_validator

CoordinateSystem: TypeAlias = Literal['pixel', 'patient']
Point3D: TypeAlias = tuple[int | float, int | float, int | float | None]


def _normalize_optional_coordinate(value: Any) -> int | float | None:
    if isinstance(value, np.generic):
        value = value.item()

    if value is None:
        return None

    if not isinstance(value, (int, float)):
        raise TypeError(f'Point coordinates must be numeric, got {type(value)}.')

    return value


def _normalize_required_coordinate(value: Any) -> int | float:
    normalized = _normalize_optional_coordinate(value)
    if normalized is None:
        raise TypeError('Point coordinates cannot be None.')
    return normalized


def _normalize_point(point: Any, *, allow_2d: bool = True) -> Point3D:
    if isinstance(point, np.ndarray):
        point = point.tolist()

    if not isinstance(point, (list, tuple)):
        raise TypeError(f'Points must be tuples, lists, or numpy arrays, got {type(point)}.')

    if len(point) == 2 and allow_2d:
        x, y = point
        return (
            _normalize_required_coordinate(x),
            _normalize_required_coordinate(y),
            None,
        )

    if len(point) == 3:
        x, y, z = point
        return (
            _normalize_required_coordinate(x),
            _normalize_required_coordinate(y),
            _normalize_optional_coordinate(z),
        )

    expected = '2 or 3' if allow_2d else '3'
    raise ValueError(f'Points must contain {expected} coordinates, got {len(point)}.')


class Geometry(BaseModel):
    """Base geometry payload for annotation entities."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={'type'}, warnings='none')
        if 'points' in payload:
            payload['points'] = [list(point) for point in payload['points']]
        return payload


class _TwoPointGeometry(Geometry):
    points: tuple[Point3D, Point3D]

    @field_validator('points', mode='before')
    @classmethod
    def _validate_points(cls, value: Any) -> tuple[Point3D, Point3D]:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError('Two-point geometries require exactly two points.')

        point1, point2 = value
        return (
            _normalize_point(point1),
            _normalize_point(point2),
        )

    @property
    def point1(self) -> Point3D:
        return self.points[0]

    @property
    def point2(self) -> Point3D:
        return self.points[1]

    @classmethod
    def from_coordinates(
        cls,
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        *,
        coords_system: CoordinateSystem = 'pixel',
        frame_index: int | None = None,
        dicom_metadata: pydicom.Dataset | str | Path | None = None,
    ) -> '_TwoPointGeometry':
        if coords_system == 'pixel':
            return cls._from_pixel_coordinates(
                point1,
                point2,
                frame_index=frame_index,
                dicom_metadata=dicom_metadata,
            )

        if coords_system == 'patient':
            return cls(points=(
                _normalize_point(point1, allow_2d=False),
                _normalize_point(point2, allow_2d=False),
            ))

        raise ValueError(f'Unknown coordinate system: {coords_system}')

    @classmethod
    def _from_pixel_coordinates(
        cls,
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        *,
        frame_index: int | None = None,
        dicom_metadata: pydicom.Dataset | str | Path | None = None,
    ) -> '_TwoPointGeometry':
        normalized_point1 = _normalize_point(point1)
        normalized_point2 = _normalize_point(point2)

        if dicom_metadata is not None:
            if isinstance(dicom_metadata, (str, Path)):
                dicom_metadata = pydicom.dcmread(str(dicom_metadata))

            patient_point1 = pixel_to_patient(
                dicom_metadata,
                normalized_point1[0],
                normalized_point1[1],
                slice_index=frame_index,
            )
            patient_point2 = pixel_to_patient(
                dicom_metadata,
                normalized_point2[0],
                normalized_point2[1],
                slice_index=frame_index,
            )
            return cls(points=(
                _normalize_point(patient_point1, allow_2d=False),
                _normalize_point(patient_point2, allow_2d=False),
            ))

        z_index = frame_index
        return cls(points=(
            (normalized_point1[0], normalized_point1[1], z_index if z_index is not None else normalized_point1[2]),
            (normalized_point2[0], normalized_point2[1], z_index if z_index is not None else normalized_point2[2]),
        ))


class LineGeometry(_TwoPointGeometry):
    type: Literal['line'] = 'line'


class BoxGeometry(_TwoPointGeometry):
    type: Literal['square'] = 'square'