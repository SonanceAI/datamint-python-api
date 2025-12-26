# filepath: datamint/entities/annotation.py
"""Annotation entity module for DataMint API.

This module defines the Annotation model used to represent annotation
records returned by the DataMint API.
"""

from typing import TYPE_CHECKING, Any
import logging
import os

from ..base_entity import BaseEntity, MISSING_FIELD
from ..cache_manager import CacheManager
from pydantic import PrivateAttr
from datetime import datetime
from datamint.api.dto import AnnotationType
from datamint.types import ImagingData


if TYPE_CHECKING:
    from datamint.api.endpoints.annotations_api import AnnotationsApi
    from ..resource import Resource

logger = logging.getLogger(__name__)

# Map API field names to class attributes
_FIELD_MAPPING = {
    'type': 'annotation_type',
    'name': 'identifier',
    'added_by': 'created_by',
    'index': 'frame_index',
}

_ANNOTATION_CACHE_KEY = "annotation_data"


class AnnotationBase(BaseEntity):
    """Minimal base class for creating annotations.
    
    This class contains only the essential fields needed to create annotations.
    Use this for creating specific annotation types like ImageClassification.
    """
    
    identifier: str
    scope: str
    annotation_type: AnnotationType
    confiability: float = 1.0
    
    def __init__(self, **data):
        """Initialize the annotation base entity."""
        super().__init__(**data)

    @property
    def name(self) -> str:
        """Get the annotation name (alias for identifier)."""
        return self.identifier

class Annotation(AnnotationBase):
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

    id: str | None = None
    identifier: str
    scope: str
    frame_index: int | None = None
    text_value: str | None = None
    numeric_value: float | int | None = None
    units: str | None = None
    geometry: list | dict | None = None
    created_at: str | None = None  # ISO timestamp string
    created_by: str | None = None
    annotation_worklist_id: str | None = None
    status: str | None = None
    approved_at: str | None = None  # ISO timestamp string
    approved_by: str | None = None
    resource_id: str | None = None
    associated_file: str | None = None
    deleted: bool = False
    deleted_at: str | None = None  # ISO timestamp string
    deleted_by: str | None = None
    created_by_model: str | None = None
    set_name: str | None = None
    resource_filename: str | None = None
    resource_modality: str | None = None
    annotation_worklist_name: str | None = None
    user_info: dict | None = None
    values: list | None = MISSING_FIELD
    file: str | None = None

    _api: 'AnnotationsApi' = PrivateAttr()

    def __init__(self, **data):
        """Initialize the annotation entity."""
        super().__init__(**data)
        self._resource: 'Resource | None' = None

    @property
    def _cache(self) -> CacheManager[bytes]:
        if not hasattr(self, '__cache'):
            self.__cache = CacheManager[bytes]('annotations')
        return self.__cache

    @property
    def resource(self) -> 'Resource':
        """Lazily load and cache the associated Resource entity."""
        if self._resource is None:
            self._resource = self._api._get_resource(self)
        return self._resource

    def fetch_file_data(
        self,
        save_path: os.PathLike | str | None = None,
        auto_convert: bool = True,
        use_cache: bool = False,
    ) -> bytes | ImagingData:
        # Version info for cache validation
        version_info = self._generate_version_info()

        # Try to get from cache
        img_data = None
        if use_cache:
            img_data = self._cache.get(self.id, _ANNOTATION_CACHE_KEY, version_info)

        if img_data is None:
            # Fetch from server using download_resource_file
            logger.debug(f"Fetching image data from server for resource {self.id}")
            img_data = self._api.download_file(
                self,
                fpath_out=save_path
            )
            # Cache the data
            if use_cache:
                self._cache.set(self.id, _ANNOTATION_CACHE_KEY, img_data, version_info)

        if auto_convert:
            return self._api.convert_format(img_data)

        return img_data

    def _generate_version_info(self) -> dict:
        """Helper to generate version info for caching."""
        return {
            'created_at': self.created_at,
            'deleted_at': self.deleted_at,
            'associated_file': self.associated_file,
        }

    def invalidate_cache(self) -> None:
        """Invalidate all cached data for this annotation."""
        self._cache.invalidate(self.id)
        self._resource = None
        logger.debug(f"Invalidated cache for annotation {self.id}")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Annotation':
        """Create an Annotation instance from a dictionary.

        Args:
            data: Dictionary containing annotation data from API

        Returns:
            Annotation instance
        """
        # Convert field names and filter valid fields
        converted_data = {}
        for key, value in data.items():
            # Map field names if needed
            mapped_key = _FIELD_MAPPING.get(key, key)
            converted_data[mapped_key] = value

        if 'scope' not in converted_data:
            converted_data['scope'] = 'image' if converted_data.get('frame_index') is None else 'frame'

        if converted_data['annotation_type'] in ['segmentation']:
            if converted_data.get('file') is None:
                raise ValueError(f"Segmentation annotations must have an associated file. {data}")

        # Create instance with only valid fields
        valid_fields = {f for f in cls.model_fields.keys()}
        filtered_data = {k: v for k, v in converted_data.items() if k in valid_fields}

        return cls(**filtered_data)

    @property
    def type(self) -> str:
        """Alias for :attr:`annotation_type`."""
        return self.annotation_type

    @property
    def index(self) -> int | None:
        """Get the frame index (alias for frame_index)."""
        return self.frame_index

    @property
    def value(self) -> str | None:
        """Get the annotation value (for category annotations)."""
        return self.text_value

    @property
    def added_by(self) -> str:
        """Get the creator email (alias for created_by)."""
        return self.created_by

    def is_segmentation(self) -> bool:
        """Check if this is a segmentation annotation."""
        return self.annotation_type == 'segmentation'

    def is_label(self) -> bool:
        """Check if this is a label annotation."""
        return self.annotation_type == 'label'

    def is_category(self) -> bool:
        """Check if this is a category annotation."""
        return self.annotation_type == 'category'

    def is_frame_scoped(self) -> bool:
        """Check if this annotation is frame-scoped."""
        return self.scope == 'frame'

    def is_image_scoped(self) -> bool:
        """Check if this annotation is image-scoped."""
        return self.scope == 'image'

    def get_created_datetime(self) -> datetime | None:
        """
        Get the creation datetime as a datetime object.

        Returns:
            datetime object or None if created_at is not set
        """
        if isinstance(self.created_at, datetime):
            return self.created_at

        if self.created_at:
            try:
                return datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Could not parse created_at datetime: {self.created_at}")
        return None
