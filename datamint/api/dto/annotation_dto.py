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

from typing import Any, TYPE_CHECKING
from datamint.entities.annotations import AnnotationType

if TYPE_CHECKING:
    from datamint.entities.annotations.geometry import Geometry

def _remove_none(d: dict) -> dict:
    return {k: _remove_none(v) for k, v in d.items() if v is not None} if isinstance(d, dict) else d


class CreateAnnotationDto:
    def __init__(self,
                 type: AnnotationType | str,
                 identifier: str,
                 scope: str,
                 annotation_worklist_id: str | None = None,
                 value=None,
                 imported_from: str | None = None,
                 import_author: str | None = None,
                 frame_index: int | None = None,
                 is_model: bool | None = None,
                 model_id: str | None = None,
                 geometry: 'Geometry | None' = None,
                 units: str | None = None):
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
            if not isinstance(model_id, str):
                raise ValueError("model_id must be a string if provided.")
            is_model = True
        self.is_model = is_model
        self.geometry = geometry

        if geometry is not None and self.type != geometry.type:
            raise ValueError(f"Annotation type {self.type} does not match geometry type {geometry.type}.")

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
