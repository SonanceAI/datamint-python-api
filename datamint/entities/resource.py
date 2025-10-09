"""Resource entity module for DataMint API."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional, Any, Sequence
import logging

from .base_entity import BaseEntity, MISSING_FIELD
from .cache_manager import CacheManager
from pydantic import PrivateAttr
from datamint.api.dto import AnnotationType
import webbrowser
from datamint.types import ImagingData

if TYPE_CHECKING:
    from datamint.api.endpoints.resources_api import ResourcesApi
    from .project import Project
    from .annotation import Annotation

logger = logging.getLogger(__name__)


_IMAGE_CACHEKEY = "image_data"


class Resource(BaseEntity):
    """Represents a DataMint resource with all its properties and metadata.

    This class models a resource entity from the DataMint API, containing
    information about uploaded files, their metadata, and associated projects.

    Attributes:
        id: Unique identifier for the resource
        resource_uri: URI path to access the resource file
        storage: Storage type (e.g., 'DicomResource')
        location: Storage location path
        upload_channel: Channel used for upload (e.g., 'tmp')
        filename: Original filename of the resource
        modality: Medical imaging modality
        mimetype: MIME type of the file
        size: File size in bytes
        upload_mechanism: Mechanism used for upload (e.g., 'api')
        customer_id: Customer/organization identifier
        status: Current status of the resource
        created_at: ISO timestamp when resource was created
        created_by: Email of the user who created the resource
        published: Whether the resource is published
        published_on: ISO timestamp when resource was published
        published_by: Email of the user who published the resource
        publish_transforms: Optional publication transforms
        deleted: Whether the resource is deleted
        deleted_at: Optional ISO timestamp when resource was deleted
        deleted_by: Optional email of the user who deleted the resource
        metadata: Resource metadata with DICOM information
        source_filepath: Original source file path
        tags: List of tags associated with the resource
        instance_uid: DICOM SOP Instance UID (top-level)
        series_uid: DICOM Series Instance UID (top-level)
        study_uid: DICOM Study Instance UID (top-level)
        patient_id: Patient identifier (top-level)
        segmentations: Optional segmentation data
        measurements: Optional measurement data
        categories: Optional category data
        labels: List of labels associated with the resource
        user_info: Information about the user who created the resource
        projects: List of projects this resource belongs to
    """
    id: str
    resource_uri: str
    storage: str
    location: str
    upload_channel: str
    filename: str
    modality: str
    mimetype: str
    size: int
    upload_mechanism: str
    customer_id: str
    status: str
    created_at: str
    created_by: str
    published: bool
    deleted: bool
    source_filepath: str | None
    metadata: dict
    projects: list[dict] = MISSING_FIELD
    published_on: str | None
    published_by: str | None
    tags: list[str] | None = None
    publish_transforms: Optional[Any] = None
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    instance_uid: Optional[str] = None
    series_uid: Optional[str] = None
    study_uid: Optional[str] = None
    patient_id: Optional[str] = None
    segmentations: Optional[Any] = None  # TODO: Define proper type when spec available
    measurements: Optional[Any] = None  # TODO: Define proper type when spec available
    categories: Optional[Any] = None  # TODO: Define proper type when spec available
    user_info: Optional[dict] = None

    _api: 'ResourcesApi' = PrivateAttr()

    def __init__(self, **data):
        """Initialize the resource entity."""
        super().__init__(**data)
        self._cache: CacheManager[bytes] = CacheManager[bytes]('resources')

    def fetch_file_data(
        self,
        auto_convert: bool = True,
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> bytes | ImagingData:
        """Get the file data for this resource.

        This method automatically caches the file data locally. On subsequent
        calls, it checks the server for changes and uses cached data if unchanged.

        Args:
            use_cache: If True, uses cached data when available and valid
            auto_convert: If True, automatically converts to appropriate format (pydicom.Dataset, PIL Image, etc.)
            save_path: Optional path to save the file locally

        Returns:
            File data (format depends on auto_convert and file type)
        """
        # Version info for cache validation
        version_info = self._generate_version_info()

        # Try to get from cache
        img_data = None
        if use_cache:
            img_data = self._cache.get(self.id, _IMAGE_CACHEKEY, version_info)
            if img_data is not None:
                logger.debug(f"Using cached image data for resource {self.id}")

        if img_data is None:
            # Fetch from server using download_resource_file
            logger.debug(f"Fetching image data from server for resource {self.id}")
            img_data = self._api.download_resource_file(
                self,
                save_path=save_path,
                auto_convert=False
            )
            # Cache the data
            if use_cache:
                self._cache.set(self.id, _IMAGE_CACHEKEY, img_data, version_info)

        if auto_convert:
            try:
                mimetype, ext = self._api._determine_mimetype(img_data, self)
                img_data = self._api.convert_format(img_data,
                                                    mimetype=mimetype,
                                                    file_path=save_path)
            except Exception as e:
                logger.error(f"Failed to auto-convert resource {self.id}: {e}")

        return img_data

    def _generate_version_info(self) -> dict:
        """Helper to generate version info for caching."""
        return {
            'created_at': self.created_at,
            'deleted_at': self.deleted_at,
            'size': self.size,
        }

    def _save_into_cache(self, data: bytes) -> None:
        """Helper to save raw data into cache."""
        version_info = self._generate_version_info()
        self._cache.set(self.id, _IMAGE_CACHEKEY, data, version_info)

    def fetch_annotations(
        self,
        annotation_type: AnnotationType | str | None = None
    ) -> Sequence['Annotation']:
        """Get annotations associated with this resource."""

        annotations = self._api.get_annotations(self)

        if annotation_type:
            annotation_type = AnnotationType(annotation_type)
            annotations = [a for a in annotations if a.annotation_type == annotation_type]
        return annotations

    # def get_projects(
    #     self,
    # ) -> Sequence['Project']:
    #     """Get all projects this resource belongs to.

    #     Returns:
    #         List of Project instances
    #     """
    #     return self._api.get_projects(self)

        
    def invalidate_cache(self) -> None:
        """Invalidate cached data for this resource.
        """
        # Invalidate all
        self._cache.invalidate(self.id)
        logger.debug(f"Invalidated all cache for resource {self.id}")

    @property
    def size_mb(self) -> float:
        """Get file size in megabytes.

        Returns:
            File size in MB rounded to 2 decimal places
        """
        return round(self.size / (1024 * 1024), 2)

    def is_dicom(self) -> bool:
        """Check if the resource is a DICOM file.

        Returns:
            True if the resource is a DICOM file, False otherwise
        """
        return self.mimetype == 'application/dicom' or self.storage == 'DicomResource'

    # def get_project_names(self) -> list[str]:
    #     """Get list of project names this resource belongs to.

    #     Returns:
    #         List of project names
    #     """
    #     return [proj['name'] for proj in self.projects] if self.projects != MISSING_FIELD else []

    def __str__(self) -> str:
        """String representation of the resource.

        Returns:
            Human-readable string describing the resource
        """
        return f"Resource(id='{self.id}', filename='{self.filename}', size={self.size_mb}MB)"

    def __repr__(self) -> str:
        """Detailed string representation of the resource.

        Returns:
            Detailed string representation for debugging
        """
        return (
            f"Resource(id='{self.id}', filename='{self.filename}', "
            f"modality='{self.modality}', status='{self.status}', "
            f"published={self.published})"
        )
    
    @property
    def url(self) -> str:
        """Get the URL to access this resource in the DataMint web application."""
        base_url = self._api.config.web_app_url
        return f'{base_url}/resource/{self.id}'

    def show(self) -> None:
        """Open the resource in the default web browser."""
        webbrowser.open(self.url)
