from __future__ import annotations
from typing import Any, ClassVar, Literal, TypeAlias
import logging
from nibabel.nifti1 import Nifti1Image
import numpy as np
from medimgkit.dicom_utils import get_slice_orientation
import pydicom
from medimgkit import ViewPlane, dicom_utils, nifti_utils
from pydantic import BaseModel, ConfigDict, field_validator

CoordinateSystem: TypeAlias = Literal['pixel', 'patient']
Point3D: TypeAlias = tuple[int | float, int | float, int | float]
Vector3D: TypeAlias = tuple[float, float, float]

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
            0,
        )

    if len(point) == 3:
        x, y, z = point
        return (
            _normalize_required_coordinate(x),
            _normalize_required_coordinate(y),
            _normalize_required_coordinate(z),
        )

    expected = '2 or 3' if allow_2d else '3'
    raise ValueError(f'Points must contain {expected} coordinates, got {len(point)}.')


def _normalize_vector(vector: np.ndarray | list | tuple | None) -> Vector3D | None:
    if vector is None:
        return None

    if isinstance(vector, np.ndarray):
        vector = vector.tolist()

    if not isinstance(vector, (list, tuple)) or len(vector) != 3:
        raise TypeError(f'Expected a 3D vector, got {type(vector)}.')

    x, y, z = vector
    return (
        float(_normalize_required_coordinate(x)),
        float(_normalize_required_coordinate(y)),
        float(_normalize_required_coordinate(z)),
    )


class Geometry(BaseModel):
    """Base geometry payload for annotation entities."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='ignore')

    type: ClassVar[str]
    viewPlaneNormal: Vector3D | None = None
    viewUp: Vector3D | None = None
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
            origin = nifti_utils.pixel_to_world(metadata, pixel_x=0, pixel_y=0, slice_index=0)
            view_right = nifti_utils.pixel_to_world(metadata, pixel_x=1, pixel_y=0, slice_index=0) - origin
            viewUp = nifti_utils.pixel_to_world(metadata, pixel_x=0, pixel_y=1, slice_index=0) - origin

            # IMPORTANT: Datamint assumes opposite direction of slices. So multiply by -1.
            viewPlaneNormal = -np.cross(view_right, viewUp)
            viewPlaneNormal /= np.linalg.norm(viewPlaneNormal)
            viewUp /= np.linalg.norm(viewUp)
            return _normalize_vector(viewPlaneNormal), _normalize_vector(viewUp)
        elif isinstance(metadata, pydicom.Dataset):
            slice_orient = get_slice_orientation(metadata, 0)
            viewPlaneNormal = slice_orient / np.linalg.norm(slice_orient)
            # the second direction vector in the image orientation corresponds to the "up" direction (column direction) in the view
            viewUp = dicom_utils.get_image_orientation(metadata, slice_index=0)[3:6]
            viewUp = viewUp / np.linalg.norm(viewUp)
            return _normalize_vector(viewPlaneNormal), _normalize_vector(viewUp)
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
                                                        slice_index=frame_index,
                                                        plane=slice_plane)
            # patient_point1.shape: (3,)
            patient_point2 = nifti_utils.pixel_to_world(metadata,
                                                        pixel_x=point2[0],
                                                        pixel_y=point2[1],
                                                        slice_index=frame_index,
                                                        plane=slice_plane)
            # IMPORTANT: Datamint assumes opposite direction of slices. So multiply by -1.
            patient_point1[:2] = -patient_point1[:2]
            patient_point2[:2] = -patient_point2[:2]
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
    type: ClassVar[str] = 'line'


class BoxGeometry(Geometry):
    points: tuple[Point3D, Point3D, Point3D, Point3D]
    type: ClassVar[str] = 'square'

    @field_validator('points', mode='before')
    @classmethod
    def _validate_points(cls, value: Any) -> tuple[Point3D, Point3D, Point3D, Point3D]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ValueError('Box geometries require exactly four corner points.')

        point1, point2, point3, point4 = value
        return (
            _normalize_point(point1, allow_2d=False),
            _normalize_point(point2, allow_2d=False),
            _normalize_point(point3, allow_2d=False),
            _normalize_point(point4, allow_2d=False),
        )

    @property
    def point1(self) -> Point3D:
        return self.points[0]

    @property
    def point2(self) -> Point3D:
        return self.points[3]

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
    ) -> 'BoxGeometry':
        if coords_system == 'pixel':
            return cls._from_pixel_coordinates(
                point1,
                point2,
                frame_index=frame_index,
                metadata=metadata,
                slice_plane=slice_plane,
            )

        if coords_system == 'patient':
            normalized_point1 = _normalize_point(point1, allow_2d=False)
            normalized_point2 = _normalize_point(point2, allow_2d=False)
            viewPlaneNormal, viewUp = _TwoPointGeometry._extract_view_parameters(metadata)
            points = cls._box_points_from_corners(
                normalized_point1,
                normalized_point2,
                viewPlaneNormal=viewPlaneNormal,
                viewUp=viewUp,
            )
            return cls(
                points=points,
                coordinate_system='patient',
                viewPlaneNormal=viewPlaneNormal,
                viewUp=viewUp,
            )

        raise ValueError(f'Unknown coordinate system: {coords_system}')

    @staticmethod
    def _box_points_from_corners(
        point1: Point3D,
        point2: Point3D,
        *,
        viewPlaneNormal: Vector3D | None = None,
        viewUp: Vector3D | None = None,
    ) -> tuple[Point3D, Point3D, Point3D, Point3D]:
        if viewPlaneNormal is None or viewUp is None:
            z = point1[2]
            return (
                (point1[0], point1[1], z),
                (point2[0], point1[1], z),
                (point1[0], point2[1], z),
                (point2[0], point2[1], z),
            )

        view_right = np.cross(np.asarray(viewPlaneNormal, dtype=float), np.asarray(viewUp, dtype=float))
        view_right_norm = np.linalg.norm(view_right)
        if view_right_norm == 0:
            raise ValueError('Could not determine box orientation from metadata.')
        view_right = view_right / view_right_norm

        normalized_view_up = np.asarray(viewUp, dtype=float)
        view_up_norm = np.linalg.norm(normalized_view_up)
        if view_up_norm == 0:
            raise ValueError('Could not determine box orientation from metadata.')
        normalized_view_up = normalized_view_up / view_up_norm
        view_down = -normalized_view_up

        point1_arr = np.asarray(point1, dtype=float)
        point2_arr = np.asarray(point2, dtype=float)
        delta = point2_arr - point1_arr
        width = float(np.dot(delta, view_right))
        height = float(np.dot(delta, view_down))

        top_right = point1_arr + width * view_right
        bottom_left = point1_arr + height * view_down

        return (
            _normalize_point(point1_arr, allow_2d=False),
            _normalize_point(top_right, allow_2d=False),
            _normalize_point(bottom_left, allow_2d=False),
            _normalize_point(point2_arr, allow_2d=False),
        )

    @classmethod
    def _from_pixel_coordinates(
        cls,
        point1: tuple[int, int] | tuple[float, float, float],
        point2: tuple[int, int] | tuple[float, float, float],
        *,
        frame_index: int | None = None,
        slice_plane: ViewPlane | None = None,
        metadata: pydicom.Dataset | Nifti1Image | None = None,
    ) -> 'BoxGeometry':
        normalized_point1 = _normalize_point(point1)
        normalized_point2 = _normalize_point(point2)

        viewPlaneNormal, viewUp = _TwoPointGeometry._extract_view_parameters(metadata)

        if metadata is not None:
            patient_point1, patient_point2 = _TwoPointGeometry._pixel_to_patient_coordinates(
                normalized_point1,
                normalized_point2,
                frame_index=frame_index,
                slice_plane=slice_plane,
                metadata=metadata,
            )
            points = cls._box_points_from_corners(
                _normalize_point(patient_point1, allow_2d=False),
                _normalize_point(patient_point2, allow_2d=False),
                viewPlaneNormal=viewPlaneNormal,
                viewUp=viewUp,
            )
            return cls(
                points=points,
                coordinate_system='patient',
                viewPlaneNormal=viewPlaneNormal,
                viewUp=viewUp,
            )

        _LOGGER.warning('No metadata provided for pixel to patient coordinate conversion;'
                        ' This is not recommended as the coordinates might be wrongly interpreted')

        z_index = frame_index
        pixel_point1 = (
            normalized_point1[0],
            normalized_point1[1],
            z_index if z_index is not None else normalized_point1[2],
        )
        pixel_point2 = (
            normalized_point2[0],
            normalized_point2[1],
            z_index if z_index is not None else normalized_point2[2],
        )
        return cls(
            points=cls._box_points_from_corners(pixel_point1, pixel_point2),
            coordinate_system='pixel',
        )
