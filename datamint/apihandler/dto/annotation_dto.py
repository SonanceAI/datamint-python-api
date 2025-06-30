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
from typing import Any
import logging
from enum import Enum


_LOGGER = logging.getLogger(__name__)


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

    @staticmethod
    def map_cornerstone(self) -> dict['AnnotationType', str]:
        return {
            # AnnotationType.SEGMENTATION: "cornerstone",
            # AnnotationType.AREA: "cornerstone",
            # AnnotationType.DISTANCE: "cornerstone",
            # AnnotationType.ANGLE: "cornerstone",
            # AnnotationType.POINT: "cornerstone",
            AnnotationType.LINE: "Length",
            # AnnotationType.REGION: "cornerstone",
            AnnotationType.SQUARE: "RectangleROI",
            # AnnotationType.CIRCLE: "cornerstone",
            # AnnotationType.CATEGORY: "cornerstone",
            # AnnotationType.LABEL: "cornerstone"
        }

    def cornerstone_equivalent(self) -> str:
        """
        Returns the equivalent cornerstone annotation type for the current type.
        """
        mapping = self.map_cornerstone()
        if self in mapping:
            return mapping[self]
        else:
            raise ValueError(f"No equivalent cornerstone type for {self}.")


def _remove_none(d: dict) -> dict:
    return {k: _remove_none(v) for k, v in d.items() if v is not None} if isinstance(d, dict) else d


class Box:
    def __init__(self, x0, y0, x1, y1, frame_index):
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.frame_index = frame_index


class Handles:
    def __init__(self, points: list[list[float]]):
        self.points = points

    def to_dict(self) -> dict:
        return {
            "points": self.points,
            "textBox": None,
            "activeHandleIndex": None,
            "textBox": {
                "hasMoved": False,
            },
            "activeHandleIndex": None,
        }


class ExternalDescription:
    class Metadata:
        def __init__(self, sliceIndex: int,
                     referencedImageId: str,
                     toolName: str,
                     viewUp: list[int] = None,
                     cameraPosition: list[float] = None,
                     viewPlaneNormal: list[int] = None, cameraFocalPoint: list[float] = None,
                     FrameOfReferenceUID: str = None,
                     ):
            self.viewUp = viewUp
            self.toolName = toolName
            self.sliceIndex = sliceIndex
            self.cameraPosition = cameraPosition
            self.viewPlaneNormal = viewPlaneNormal
            self.cameraFocalPoint = cameraFocalPoint
            self.referencedImageId = referencedImageId
            self.FrameOfReferenceUID = FrameOfReferenceUID

            if self.referencedImageId is not None:
                if self.referencedImageId.startswith("dicomfile:") and self.FrameOfReferenceUID is None:
                    _LOGGER.warning(f"FrameOfReferenceUID is None, but referencedImageId is a dicomfile. ")

            if 'frame=' not in referencedImageId:
                self.referencedImageId = referencedImageId + f"&frame={sliceIndex+1}"

        def to_dict(self) -> dict:
            return {
                "viewUp": self.viewUp,
                "toolName": self.toolName,
                "sliceIndex": self.sliceIndex,
                "cameraPosition": self.cameraPosition,
                "viewPlaneNormal": self.viewPlaneNormal,
                "cameraFocalPoint": self.cameraFocalPoint,
                "referencedImageId": self.referencedImageId,
                "FrameOfReferenceUID": self.FrameOfReferenceUID
            }

    def __init__(self,
                 annotationUID: str,
                 label: str, handles: Handles,
                 metadata: Metadata,
                 highlighted: bool = False, invalidated: bool = False, isLocked: bool = False,
                 isVisible: bool = True,):
        self.annotationUID = annotationUID
        self.data = {
            "label": label,
            "handles": handles.to_dict(),
            "cachedStats": {}
        }
        self.highlighted = highlighted
        self.invalidated = invalidated
        self.isLocked = isLocked
        self.isVisible = isVisible
        self.metadata = metadata

    def to_dict(self) -> dict:
        return {
            "data": self.data,
            "highlighted": self.highlighted,
            "invalidated": self.invalidated,
            "isLocked": self.isLocked,
            "isVisible": self.isVisible,
            "metadata": self.metadata.to_dict(),
        }


class SamGeometry:
    def __init__(self, boxes: list[Box] | None = None, points=None,
                 ):
        self.boxes = boxes if boxes else None
        self.points = points if points else None

    def to_dict(self) -> dict:
        return {
            "boxes": [box.__dict__ for box in self.boxes] if self.boxes else None,
            "points": [point.__dict__ for point in self.points] if self.points else None,
        }


class MainGeometry:
    def __init__(self,
                 sam_geometry: SamGeometry,
                 external_description: ExternalDescription):
        self.sam_geometry = sam_geometry
        self.external_description = external_description

    def to_dict(self) -> dict:
        sam_geometry_dict = self.sam_geometry.to_dict() if self.sam_geometry else None
        ret = {
            "external_description": self.external_description.to_dict()
        }
        if sam_geometry_dict:
            for key in sam_geometry_dict:
                ret[key] = sam_geometry_dict[key]
        return ret


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
                 geometry: MainGeometry | None = None,
                 units: str = None):
        self.value = value
        self.type = type if isinstance(type, AnnotationType) else AnnotationType(type)
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

        if isinstance(geometry, dict):
            self.geometry = MainGeometry(**geometry)
        else:
            self.geometry = geometry

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
            "geometry": self.geometry.to_dict() if self.geometry else None
        }
        return _remove_none(ret)
