# filepath: datamint/entities/annotation.py
"""Annotation entity module for DataMint API.

This module defines the Annotation model used to represent annotation
records returned by the DataMint API.
"""

from typing import Any
import logging
from .base_entity import BaseEntity

logger = logging.getLogger(__name__)


class Annotation(BaseEntity):
    """Pydantic Model representing a DataMint annotation.

    Attributes:
        id: Unique identifier for the annotation.
        identifier: User-friendly identifier or label for the annotation.
        scope: Scope of the annotation (e.g., "frame", "image").
        frame_index: Index of the frame if scope is frame-based.
        annotation_type: Type of annotation (e.g., "segmentation", "bbox", "label").
        text_value: Optional text value associated with the annotation.
        numeric_value: Optional numeric value associated with the annotation.
        units: Optional units for numeric_value.
        geometry: Optional geometry payload (e.g., polygons, masks) as a list.
        created_at: ISO timestamp for when the annotation was created.
        created_by: Email or identifier of the creating user.
        annotation_worklist_id: Optional worklist ID associated with the annotation.
        status: Lifecycle status of the annotation (e.g., "new", "approved").
        approved_at: Optional ISO timestamp for approval time.
        approved_by: Optional identifier of the approver.
        resource_id: ID of the resource this annotation belongs to.
        associated_file: Path or identifier of any associated file artifact.
        deleted: Whether the annotation is marked as deleted.
        deleted_at: Optional ISO timestamp for deletion time.
        deleted_by: Optional identifier of the user who deleted the annotation.
        created_by_model: Optional identifier of the model that created this annotation.
        old_geometry: Optional previous geometry payload for change tracking.
        set_name: Optional set name this annotation belongs to.
        resource_filename: Optional filename of the resource.
        resource_modality: Optional modality of the resource (e.g., CT, MR).
        annotation_worklist_name: Optional worklist name associated with the annotation.
        user_info: Optional user information with keys like firstname and lastname.
        values: Optional extra values payload for flexible schemas.
    """

    id: str
    identifier: str
    scope: str
    frame_index: int
    annotation_type: str
    text_value: str
    numeric_value: float | int
    units: str
    geometry: list
    created_at: str  # ISO timestamp string
    created_by: str
    annotation_worklist_id: str
    status: str
    approved_at: str  # ISO timestamp string
    approved_by: str
    resource_id: str
    associated_file: str
    deleted: bool
    deleted_at: str  # ISO timestamp string
    deleted_by: str
    created_by_model: str
    old_geometry: Any
    set_name: str
    resource_filename: str
    resource_modality: str
    annotation_worklist_name: str
    user_info: dict
    values: Any

    # TODO: Consider constraining some fields with Literal types and parsing timestamps to datetime
    #       once the API schema is stable, to provide stronger validation.
