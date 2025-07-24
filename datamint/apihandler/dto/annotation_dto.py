"""
Data Transfer Objects (DTOs) for handling annotations in the datamint-python-api.

This module provides classes for creating and manipulating annotation data
that can be sent to or received from the Datamint API. It includes structures
for different annotation geometry types, metadata, and formatting utilities.

Classes:
    Handles (cornerstone): Manages annotation control points and handle properties.
    ExternalDescription (cornerstone): Contains external metadata for annotations.
        Metadata (cornerstone): Nested class for managing annotation positioning and reference metadata.
    SamGeometry (datamint): Represents Segment Anything Model geometry with boxes and points.
    MainGeometry: Combines SAM geometry with external descriptions.
    CreateAnnotationDto: Main DTO for creating annotation requests.
"""

import json
from typing import Any, TypeAlias, Literal
import logging
from enum import Enum
from medimgkit.dicom_utils import pixel_to_patient
import pydicom
import numpy as np


_LOGGER = logging.getLogger(__name__)


CoordinateSystem: TypeAlias = Literal['pixel', 'patient']
"""TypeAlias: The available coordinate systems for annotation geometry. Possible values are 'pixel' and 'patient' (used in DICOMs).
"""


class AnnotationType(Enum):
    SEGMENTATION = 'segmentation'
    AREA = 'area'
    DISTANCE = 'distance'
    ANGLE = 'angle'
    POINT = 'point'
    LINE = 'line'
    REGION = 'region'
    SQUARE = 'square'
    CIRCLE = 'circle'
    CATEGORY = 'category'
    LABEL = 'label'


def _remove_none(d: dict) -> dict:
    return {k: _remove_none(v) for k, v in d.items() if v is not None} if isinstance(d, dict) else d


class Box:
    def __init__(self, x0, y0, x1, y1, frame_index):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.frame_index = frame_index


class Geometry:
    def __init__(self, type: AnnotationType | str):
        self.type = type if isinstance(type, AnnotationType) else AnnotationType(type)

    def to_dict(self) -> dict:
        raise NotImplementedError("Subclasses must implement to_dict method.")


class LineGeometry(Geometry):
    def __init__(self, point1: tuple[float, float, float],
                 point2: tuple[float, float, float]):
        super().__init__(AnnotationType.LINE)
        if isinstance(point1, np.ndarray):
            point1 = point1.tolist()
        if isinstance(point2, np.ndarray):
            point2 = point2.tolist()
        self.point1 = point1
        self.point2 = point2

    def to_dict(self) -> dict:
        return {
            'points': [self.point1, self.point2],
        }

    @staticmethod
    def from_dicom(ds: pydicom.Dataset,
                   point1: tuple[int, int],
                   point2: tuple[int, int],
                   slice_index: int | None = None) -> 'LineGeometry':
        pixel_x1, pixel_y1 = point1
        pixel_x2, pixel_y2 = point2

        new_point1 = pixel_to_patient(ds, pixel_x1, pixel_y1,
                                      slice_index=slice_index)
        new_point2 = pixel_to_patient(ds, pixel_x2, pixel_y2,
                                      slice_index=slice_index)
        return LineGeometry(new_point1, new_point2)


class BoxGeometry(Geometry):
    def __init__(self, point1: tuple[float, float, float],
                 point2: tuple[float, float, float]):
        """
        Create a box geometry from two diagonal corner points.

        Args:
            point1: First corner point (x, y, z) or (x, y, frame_index)
            point2: Opposite diagonal corner point (x, y, z) or (x, y, frame_index)
        """
        super().__init__(AnnotationType.SQUARE)  # Using SQUARE as the box type
        if isinstance(point1, np.ndarray):
            point1 = point1.tolist()
        if isinstance(point2, np.ndarray):
            point2 = point2.tolist()
        self.point1 = point1
        self.point2 = point2

    def to_dict(self) -> dict:
        return {
            'points': [self.point1, self.point2],
        }

    @staticmethod
    def from_dicom(ds: pydicom.Dataset,
                   point1: tuple[int, int],
                   point2: tuple[int, int],
                   slice_index: int | None = None) -> 'BoxGeometry':
        """
        Create a box geometry from DICOM pixel coordinates.

        Args:
            ds: DICOM dataset containing spatial metadata
            point1: First corner in pixel coordinates (x, y)
            point2: Opposite corner in pixel coordinates (x, y)
            slice_index: The slice/frame index for 3D positioning

        Returns:
            BoxGeometry with patient coordinate points
        """
        pixel_x1, pixel_y1 = point1
        pixel_x2, pixel_y2 = point2

        new_point1 = pixel_to_patient(ds, pixel_x1, pixel_y1,
                                      slice_index=slice_index)
        new_point2 = pixel_to_patient(ds, pixel_x2, pixel_y2,
                                      slice_index=slice_index)
        return BoxGeometry(new_point1, new_point2)


class CreateAnnotationDto:
    def __init__(self,
                 type: AnnotationType | str,
                 identifier: str,
                 scope: str,
                 annotation_worklist_id: str,
                 value=None,
                 imported_from: str | None = None,
                 import_author: str | None = None,
                 frame_index: int | None = None,
                 is_model: bool = None,
                 model_id: str | None = None,
                 geometry: Geometry | None = None,
                 units: str = None):
        self.type = type if isinstance(type, AnnotationType) else AnnotationType(type)
        self.value = value
        self.identifier = identifier
        self.scope = scope
        self.annotation_worklist_id = annotation_worklist_id
        self.imported_from = imported_from
        self.import_author = import_author
        self.frame_index = frame_index
        self.units = units
        self.model_id = model_id
        if model_id is not None:
            if is_model == False:
                raise ValueError("model_id==False while self.model_id is provided.")
            is_model = True
        self.is_model = is_model
        self.geometry = geometry

        if geometry is not None and self.type != self.geometry.type:
            raise ValueError(f"Annotation type {self.type} does not match geometry type {self.geometry.type}.")

    def to_dict(self) -> dict[str, Any]:
        ret = {
            "value": self.value,
            "type": self.type.value,
            "identifier": self.identifier,
            "scope": self.scope,
            'frame_index': self.frame_index,
            'annotation_worklist_id': self.annotation_worklist_id,
            'imported_from': self.imported_from,
            'import_author': self.import_author,
            'units': self.units,
            "geometry": self.geometry.to_dict() if self.geometry else None,
            "is_model": self.is_model,
            "model_id": self.model_id
        }
        return _remove_none(ret)
