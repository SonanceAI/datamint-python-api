from __future__ import annotations
from typing import Any, Literal, TypeAlias
import logging
from nibabel.nifti1 import Nifti1Image
import numpy as np
from medimgkit.dicom_utils import get_slice_orientation
import pydicom
from medimgkit import ViewPlane, dicom_utils, nifti_utils
from pydantic import BaseModel, ConfigDict, field_validator

CoordinateSystem: TypeAlias = Literal['pixel', 'patient']
Point3D: TypeAlias = tuple[int | float, int | float, int | float | None]

_LOGGER = logging.getLogger(__name__)


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


def _normalize_point(point: np.ndarray | list | tuple, *, allow_2d: bool = True) -> Point3D:
    """
    Normalize a point input to a 3D coordinate tuple. Supports input as lists, tuples, or numpy arrays.
    """
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

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    type: str
    viewPlaneNormal: tuple[float, float, float] | None = None
    viewUp: tuple[float, float, float] | None = None
    coordinate_system: CoordinateSystem = 'patient'

    def to_dict(self) -> dict[str, Any]:
        payload = self.model_dump(exclude={'type', 'coordinate_system'}, warnings='none')
        if 'points' in payload:
            payload['points'] = [list(point) for point in payload['points']]
        # remove nones from the payload to avoid confusion on the API side
        for key in ['viewPlaneNormal', 'viewUp']:
            if payload.get(key) is None:
                payload.pop(key, None)
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
        slice_plane: ViewPlane | None = None,
        frame_index: int | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
    ) -> '_TwoPointGeometry':
        if coords_system == 'pixel':
            return cls._from_pixel_coordinates(
                point1,
                point2,
                frame_index=frame_index,
                metadata=metadata,
                slice_plane=slice_plane,
            )

        if coords_system == 'patient':
            points = (
                _normalize_point(point1, allow_2d=False),
                _normalize_point(point2, allow_2d=False)
            )
            viewPlaneNormal, viewUp = cls._extract_view_parameters(metadata)
            return cls(points=points, coordinate_system='patient',
                       viewPlaneNormal=viewPlaneNormal, viewUp=viewUp)

        raise ValueError(f'Unknown coordinate system: {coords_system}')

    @staticmethod
    def _extract_view_parameters(metadata: pydicom.Dataset | Nifti1Image | None):
        if metadata is None:
            # assumes 2d data
            return None, None
        if isinstance(metadata, Nifti1Image):
            raise NotImplementedError('View parameters extraction from NIfTI metadata is not implemented yet.')
        elif isinstance(metadata, pydicom.Dataset):
            slice_orient = get_slice_orientation(metadata, 0)
            viewPlaneNormal = slice_orient / np.linalg.norm(slice_orient)
            # the second direction vector in the image orientation corresponds to the "up" direction (column direction) in the view
            viewUp = dicom_utils.get_image_orientation(metadata, slice_index=0)[3:6]
            viewUp = viewUp / np.linalg.norm(viewUp)
            return viewPlaneNormal, viewUp
        else:
            raise TypeError(f'Unsupported metadata type: {type(metadata)}')

    @staticmethod
    def _pixel_to_patient_coordinates(
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        frame_index: int | None = None,
        slice_plane: ViewPlane | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
    ):
        if isinstance(metadata, pydicom.Dataset):
            patient_point1 = dicom_utils.pixel_to_patient(
                metadata,
                point1[0],
                point1[1],
                slice_index=frame_index,
                axis=slice_plane
            )
            patient_point2 = dicom_utils.pixel_to_patient(
                metadata,
                point2[0],
                point2[1],
                slice_index=frame_index,
                axis=slice_plane
            )

        elif isinstance(metadata, Nifti1Image):
            patient_point1 = nifti_utils.pixel_to_world(metadata,
                                                        pixel_x=point1[0],
                                                        pixel_y=point1[1],
                                                        slice_index=frame_index)
            patient_point2 = nifti_utils.pixel_to_world(metadata,
                                                        pixel_x=point2[0],
                                                        pixel_y=point2[1],
                                                        slice_index=frame_index)
        else:
            raise TypeError(f'Unsupported metadata type: {type(metadata)}')

        return patient_point1, patient_point2

    @classmethod
    def _from_pixel_coordinates(
        cls,
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        *,
        frame_index: int | None = None,
        slice_plane: ViewPlane | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
    ) -> '_TwoPointGeometry':
        point1 = _normalize_point(point1)
        point2 = _normalize_point(point2)

        viewPlaneNormal, viewUp = cls._extract_view_parameters(metadata)

        if metadata is not None:
            patient_point1, patient_point2 = cls._pixel_to_patient_coordinates(
                point1, point2,
                frame_index=frame_index,
                slice_plane=slice_plane,
                metadata=metadata
            )
            points = (
                _normalize_point(patient_point1, allow_2d=False),
                _normalize_point(patient_point2, allow_2d=False)
            )

            return cls(points=points, coordinate_system='patient',
                       viewPlaneNormal=viewPlaneNormal, viewUp=viewUp)
        else:
            _LOGGER.warning('No metadata provided for pixel to patient coordinate conversion;'
                            ' This is not recommended as the coordinates might be wrongly interpreted')

        z_index = frame_index
        points = (
            (point1[0], point1[1], z_index if z_index is not None else point1[2]),
            (point2[0], point2[1], z_index if z_index is not None else point2[2])
        )
        return cls(points=points, coordinate_system='pixel')


class LineGeometry(_TwoPointGeometry):
    type: Literal['line'] = 'line'


class BoxGeometry(_TwoPointGeometry):
    type: Literal['square'] = 'square'
