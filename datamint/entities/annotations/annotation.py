# filepath: datamint/entities/annotation.py
"""Annotation entity module for DataMint API.

This module defines the Annotation model used to represent annotation
records returned by the DataMint API.
"""

from datetime import datetime
import logging
from typing import TYPE_CHECKING, Any, Literal, overload

from pydantic import ConfigDict, Field, PrivateAttr, field_validator

from datamint.types import CacheMode, ImagingData

from ..base_entity import BaseEntity, MISSING_FIELD
from ..cache_manager import CacheManager
from .types import AnnotationType

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
    'value': 'text_value',
}

_ANNOTATION_CACHE_KEY = "annotation_data"


def _normalize_annotation_data(data: dict[str, Any]) -> dict[str, Any]:
    converted_data: dict[str, Any] = {}
    for key, value in data.items():
        mapped_key = _FIELD_MAPPING.get(key, key)
        converted_data[mapped_key] = value

    if 'scope' not in converted_data:
        converted_data['scope'] = 'image' if converted_data.get('frame_index') is None else 'frame'

    return converted_data


class AnnotationBase(BaseEntity):
    """Minimal base class for creating annotations.

    This class contains only the essential fields needed to create annotations.
    Use this for creating specific annotation types like ImageClassification.
    """

    model_config = ConfigDict(populate_by_name=True)

    identifier: str = Field(alias="name")
    scope: str
    annotation_type: str  # AnnotationType # mlflow model signature does not support enum types
    confiability: float = 1.0

    @field_validator("annotation_type", mode="before")
    @classmethod
    def _validate_annotation_type(cls, value: AnnotationType | str) -> str:
        if isinstance(value, AnnotationType):
            return value.value

        try:
            return AnnotationType(value).value
        except ValueError as exc:
            valid_types = ", ".join(member.value for member in AnnotationType)
            raise ValueError(
                f"Invalid annotation_type {value!r}. Expected one of: {valid_types}"
            ) from exc

    def __init__(self, **data):
        """Initialize the annotation base entity."""
        super().__init__(**data)

    @property
    def name(self) -> str:
        """Get the annotation name (alias for identifier)."""
        return self.identifier
    
    def is_segmentation(self) -> bool:
        """Check if this is a segmentation annotation."""
        return self.annotation_type == AnnotationType.SEGMENTATION.value

    def is_label(self) -> bool:
        """Check if this is a label annotation."""
        return self.annotation_type == AnnotationType.LABEL.value

    def is_category(self) -> bool:
        """Check if this is a category annotation."""
        return self.annotation_type == AnnotationType.CATEGORY.value


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
        geometry: Optional geometry payload (e.g., polygons) as a list.
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
    identifier: str = Field(alias="name")
    scope: str
    frame_index: int | None = None
    text_value: str | None = None
    numeric_value: float | int | None = None
    units: str | None = None
    geometry: list | dict | None = None
    created_at: str | None = None  # ISO timestamp string
    created_by: str | None = None
    annotation_worklist_id: str | None = None
    imported_from: str | None = None
    import_author: str | None = None
    status: str | None = None
    approved_at: str | None = None  # ISO timestamp string
    approved_by: str | None = None
    resource_id: str | None = None
    associated_file: str | None = None
    deleted: bool = False
    deleted_at: str | None = None  # ISO timestamp string
    deleted_by: str | None = None
    created_by_model: str | None = None
    is_model: bool | None = None
    model_id: str | None = None
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
        """Lazily load and cache the associated Resource entity.

        Example:
            >>> annotation = api.annotations.get_list(limit=1)[0]
            >>> annotation.resource.filename
        """
        if self._resource is None:
            self._resource = self._api._get_resource(self)
        return self._resource

    @overload
    def fetch_file_data(
        self,
        auto_convert: Literal[True] = True,
        save_path: str | None = None,
        use_cache: CacheMode = False,
    ) -> 'ImagingData': ...

    @overload
    def fetch_file_data(
        self,
        auto_convert: Literal[False],
        save_path: str | None = None,
        use_cache: CacheMode = False,
    ) -> bytes: ...

    def fetch_file_data(
        self,
        auto_convert: bool = True,
        save_path: str | None = None,
        use_cache: CacheMode = False,
    ) -> 'bytes | ImagingData':
        """Get the file data for this annotation.

        Args:
            save_path: Optional path to save the file locally. If
                      ``use_cache=True``, the file is saved to save_path and
                      cache metadata points to that location (no duplication -
                      only one file on disk).
            auto_convert: If True, automatically converts to appropriate format
            use_cache: Cache behavior for this call. Use ``False`` to bypass
                cache entirely, ``True`` to read from and save to cache, or
                ``"loadonly"`` to read from cache without saving cache misses.

        Returns:
            File data (format depends on auto_convert and file type)

        Example:
            >>> annotation = api.annotations.get_list(limit=1)[0]
            >>> data = annotation.fetch_file_data(use_cache=True)
            >>> data = annotation.fetch_file_data(use_cache="loadonly")
            >>> annotation.fetch_file_data(save_path="annotation_file")
        """
        # Version info for cache validation
        version_info = self._generate_version_info()

        # Download callback for the shared caching logic
        def download_callback(path: str | None) -> bytes:
            return self._api.download_file(
                self,
                fpath_out=path
            )

        # Use shared caching logic from BaseEntity
        img_data = self._fetch_and_cache_file_data(
            cache_manager=self._cache,
            data_key=_ANNOTATION_CACHE_KEY,
            version_info=version_info,
            download_callback=download_callback,
            save_path=save_path,
            use_cache=use_cache,
        )

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

    def _to_create_dto(self):
        """Convert this annotation entity into a create DTO."""
        from datamint.api.dto import CreateAnnotationDto
        geometry = self.geometry
        if geometry is not None and not hasattr(geometry, 'to_dict'):
            raise ValueError(
                'Geometry annotations must use typed geometry entities. '
                'Use LineAnnotation or BoxAnnotation instead of a raw geometry dict.'
            )

        return CreateAnnotationDto(
            type=self.annotation_type,
            identifier=self.identifier,
            scope=self.scope,
            annotation_worklist_id=self.annotation_worklist_id,
            value=self.text_value,
            imported_from=self.imported_from,
            import_author=self.import_author,
            frame_index=self.frame_index,
            is_model=self.is_model,
            model_id=self.model_id,
            geometry=geometry,
            units=self.units,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'Annotation':
        """Create an Annotation instance from a dictionary.

        Args:
            data: Dictionary containing annotation data from API

        Returns:
            Annotation instance
        """
        converted_data = _normalize_annotation_data(data)

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
