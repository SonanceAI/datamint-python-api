"""Resource entity module for DataMint API."""

from datetime import datetime
from typing import TYPE_CHECKING, Optional, Any, Sequence
import logging
import urllib.parse
import urllib.request

from .base_entity import BaseEntity, MISSING_FIELD
from .cache_manager import CacheManager
from pydantic import PrivateAttr
import webbrowser
from pathlib import Path
from datamint.api.base_api import BaseApi

if TYPE_CHECKING:
    from datamint.api.endpoints.resources_api import ResourcesApi
    from .project import Project
    from .annotations.annotation import Annotation
    from datamint.types import ImagingData
    from datamint.api.dto import AnnotationType


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
    mimetype: str
    size: int
    customer_id: str
    status: str
    created_at: str
    created_by: str
    published: bool
    deleted: bool
    upload_mechanism: str | None = None
    # metadata: dict[str,Any] = {}
    modality: str | None = None
    source_filepath: str | None = None
    # projects: list[dict[str, Any]] | None = None
    published_on: str | None = None
    published_by: str | None = None
    tags: list[str] | None = None
    # publish_transforms: dict[str, Any] | None = None
    deleted_at: Optional[str] = None
    deleted_by: Optional[str] = None
    instance_uid: Optional[str] = None
    series_uid: Optional[str] = None
    study_uid: Optional[str] = None
    patient_id: Optional[str] = None
    # segmentations: Optional[Any] = None  # TODO: Define proper type when spec available
    # measurements: Optional[Any] = None  # TODO: Define proper type when spec available
    # categories: Optional[Any] = None  # TODO: Define proper type when spec available
    user_info: dict[str, str | None] = MISSING_FIELD

    _api: 'ResourcesApi' = PrivateAttr()

    def __new__(cls, *args, **kwargs):
        if cls is Resource and ('local_filepath' in kwargs or 'raw_data' in kwargs):
            return super().__new__(LocalResource)
        return super().__new__(cls)

    def __init__(self, **data):
        """Initialize the resource entity."""
        super().__init__(**data)

    @property
    def _cache(self) -> CacheManager[bytes]:
        if not hasattr(self, '__cache'):
            self.__cache = CacheManager[bytes]('resources')
        return self.__cache

    def fetch_file_data(
        self,
        auto_convert: bool = True,
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> 'bytes | ImagingData':
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
                # Save cached data to save_path if provided
                if save_path:
                    with open(save_path, 'wb') as f:
                        f.write(img_data)

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
                mimetype, ext = BaseApi._determine_mimetype(img_data, self.mimetype)
                img_data = BaseApi.convert_format(img_data,
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

    def is_cached(self) -> bool:
        """Check if the resource's file data is already cached locally and valid.

        Returns:
            True if valid cached data exists, False otherwise.
        """
        version_info = self._generate_version_info()
        cached_data = self._cache.get(self.id, _IMAGE_CACHEKEY, version_info)
        return cached_data is not None
    
    @property
    def filepath_cached(self) -> Path | None:
        """Get the file path of the cached resource data, if available.

        Returns:
            Path to the cached file data, or None if not cached.
        """
        if self._cache is None:
            return None
        version_info = self._generate_version_info()
        path = self._cache.get_path(self.id, _IMAGE_CACHEKEY, version_info)
        return path

    def fetch_annotations(
        self,
        annotation_type: 'AnnotationType | str | None' = None
    ) -> Sequence['Annotation']:
        """Get annotations associated with this resource."""

        annotations = self._api.get_annotations(self, annotation_type=annotation_type)
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

    @staticmethod
    def from_local_file(file_path: str | Path):
        """Create a LocalResource instance from a local file path.

        Args:
            file_path: Path to the local file
        """
        return LocalResource(local_filepath=file_path)


class LocalResource(Resource):
    """Represents a local resource that hasn't been uploaded to DataMint API yet."""

    local_filepath: str | None = None
    raw_data: bytes | None = None

    @property
    def filepath_cached(self) -> str | None:
        """Get the file path of the local resource data.

        Returns:
            Path to the local file, or None if only raw data is available.
        """
        return self.local_filepath

    def __init__(self,
                 local_filepath: str | Path | None = None,
                 raw_data: bytes | None = None,
                 convert_to_bytes: bool = False,
                 **kwargs):
        """Initialize a local resource from a local file path, URL, or raw data.

        Args:
            local_filepath: Path to the local file or URL to an online image
            raw_data: Raw bytes of the file data
            convert_to_bytes: If True and local_filepath is provided, read file into raw_data
        """
        from medimgkit.format_detection import guess_type, DEFAULT_MIME_TYPE
        from medimgkit.modality_detector import detect_modality

        if raw_data is None and local_filepath is None:
            raise ValueError("Either local_filepath or raw_data must be provided.")
        if raw_data is not None and local_filepath is not None:
            raise ValueError("Only one of local_filepath or raw_data should be provided.")

        # Check if local_filepath is a URL
        if local_filepath is not None:
            local_filepath_str = str(local_filepath)
            if local_filepath_str.startswith(('http://', 'https://')):
                # Download content from URL
                logger.debug(f"Downloading resource from URL: {local_filepath_str}")
                try:
                    with urllib.request.urlopen(local_filepath_str) as response:
                        raw_data = response.read()
                        # Try to get content-type from response headers
                        content_type = response.headers.get('Content-Type', '').split(';')[0].strip()
                except Exception as e:
                    raise ValueError(f"Failed to download from URL: {local_filepath_str}") from e

                # Extract filename from URL
                parsed_url = urllib.parse.urlparse(local_filepath_str)
                url_path = urllib.parse.unquote(parsed_url.path)
                filename = Path(url_path).name if url_path else 'downloaded_file'

                # Determine mimetype
                mimetype, _ = guess_type(raw_data)
                if mimetype is None and content_type:
                    mimetype = content_type
                if mimetype is None:
                    mimetype = DEFAULT_MIME_TYPE

                default_values = {
                    'id': '',
                    'resource_uri': '',
                    'storage': '',
                    'location': local_filepath_str,
                    'upload_channel': '',
                    'filename': filename,
                    'modality': None,
                    'mimetype': mimetype,
                    'size': len(raw_data),
                    'upload_mechanism': '',
                    'customer_id': '',
                    'status': 'local',
                    'created_at': datetime.now().isoformat(),
                    'created_by': '',
                    'published': False,
                    'deleted': False,
                    'source_filepath': local_filepath_str,
                }
                new_kwargs = kwargs.copy()
                for key, value in default_values.items():
                    new_kwargs.setdefault(key, value)
                super(Resource, self).__init__(
                    local_filepath=None,
                    raw_data=raw_data,
                    **new_kwargs
                )
                return

        if convert_to_bytes and local_filepath:
            with open(local_filepath, 'rb') as f:
                raw_data = f.read()
                local_filepath = None
        if raw_data is not None:
            # import io
            if isinstance(raw_data, str):
                mimetype, _ = guess_type(raw_data.encode())
            else:
                mimetype, _ = guess_type(raw_data)
            default_values = {
                'id': '',
                'resource_uri': '',
                'storage': '',
                'location': '',
                'upload_channel': '',
                'filename': 'raw_data',
                'modality': None,
                'mimetype': mimetype if mimetype else DEFAULT_MIME_TYPE,
                'size': len(raw_data),
                'upload_mechanism': '',
                'customer_id': '',
                'status': 'local',
                'created_at': datetime.now().isoformat(),
                'created_by': '',
                'published': False,
                'deleted': False,
                'source_filepath': None,
            }
            new_kwargs = kwargs.copy()
            for key, value in default_values.items():
                new_kwargs.setdefault(key, value)
            super().__init__(
                local_filepath=None,
                raw_data=raw_data,
                **new_kwargs
            )
        elif local_filepath is not None:
            file_path = Path(local_filepath)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            mimetype, _ = guess_type(file_path)
            if mimetype is None or mimetype == DEFAULT_MIME_TYPE:
                logger.warning(f"Could not determine mimetype for file: {file_path}")
            size = file_path.stat().st_size
            created_at = datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()

            super().__init__(
                id="",
                resource_uri="",
                storage="",
                location=str(file_path),
                upload_channel="",
                filename=file_path.name,
                modality=detect_modality(file_path),
                mimetype=mimetype,
                size=size,
                upload_mechanism="",
                customer_id="",
                status="local",
                created_at=created_at,
                created_by="",
                published=False,
                deleted=False,
                source_filepath=str(file_path),
                local_filepath=str(file_path),
                raw_data=None,
            )

    def fetch_file_data(
        self, *args,
        auto_convert: bool = True,
        save_path: str | None = None,
        **kwargs,
    ) -> 'bytes | ImagingData':
        """Get the file data for this local resource.

        Args:
            auto_convert: If True, automatically converts to appropriate format (pydicom.Dataset, PIL Image, etc.)
            save_path: Optional path to save the file locally
        Returns:
            File data (format depends on auto_convert and file type)
        """
        if self.raw_data is not None:
            img_data = self.raw_data
            local_filepath = None
        else:
            local_filepath = str(self.local_filepath)
            with open(local_filepath, 'rb') as f:
                img_data = f.read()

        if save_path:
            with open(save_path, 'wb') as f:
                f.write(img_data)

        if auto_convert:
            try:
                mimetype, ext = BaseApi._determine_mimetype(img_data, self.mimetype)
                img_data = BaseApi.convert_format(img_data,
                                                  mimetype=mimetype,
                                                  file_path=local_filepath)
            except Exception as e:
                logger.error(f"Failed to auto-convert local resource: {e}")
                logger.error(e, exc_info=True)

        return img_data

    def __str__(self) -> str:
        """String representation of the local resource.

        Returns:
            Human-readable string describing the local resource
        """
        return f"LocalResource(filepath='{self.local_filepath}', size={self.size_mb}MB)"

    def __repr__(self) -> str:
        """Detailed string representation of the local resource.

        Returns:
            Detailed string representation for debugging
        """
        return (
            f"LocalResource(filepath='{self.local_filepath}', "
            f"filename='{self.filename}', modality='{self.modality}', "
            f"size={self.size_mb}MB)"
        )
