"""Resource entity module for DataMint API."""

from collections.abc import Sequence
from datetime import datetime
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, overload
import urllib.parse
import urllib.request
import webbrowser

from pydantic import PrivateAttr

from .base_entity import BaseEntity, MISSING_FIELD
from .cache_manager import CacheManager
from datamint.api.base_api import BaseApi
from datamint.types import CacheMode

if TYPE_CHECKING:
    from datamint.api.endpoints.resources_api import ResourcesApi
    from medimgkit import ViewPlane
    from .annotations.annotation import Annotation
    from .annotations import AnnotationType
    from datamint.types import ImagingData
    import numpy as np
    from .sliced_resource import SlicedVolumeResource
    from .sliced_video_resource import SlicedVideoResource


logger = logging.getLogger(__name__)


_IMAGE_CACHEKEY = "image_data"
_SPECIALIZED_RESOURCE_TYPES_IMPORTED = False


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
    resource_kind: ClassVar[str] = 'resource'
    resource_priority: ClassVar[int] = 0
    storage_aliases: ClassVar[tuple[str, ...]] = ()
    mimetypes: ClassVar[tuple[str, ...]] = ()
    mimetype_prefixes: ClassVar[tuple[str, ...]] = ()
    filename_suffixes: ClassVar[tuple[str, ...]] = ()

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
    metadata: dict[str, Any] = {}
    modality: str | None = None
    source_filepath: str | None = None
    # projects: list[dict[str, Any]] | None = None
    published_on: str | None = None
    published_by: str | None = None
    tags: list[str] | None = None
    # publish_transforms: dict[str, Any] | None = None
    deleted_at: str | None = None
    deleted_by: str | None = None
    instance_uid: str | None = None
    series_uid: str | None = None
    study_uid: str | None = None
    patient_id: str | None = None
    # segmentations: Optional[Any] = None  # TODO: Define proper type when spec available
    # measurements: Optional[Any] = None  # TODO: Define proper type when spec available
    # categories: Optional[Any] = None  # TODO: Define proper type when spec available
    user_info: dict[str, str | None] | str = MISSING_FIELD

    _api: 'ResourcesApi' = PrivateAttr()
    _shared_cache: ClassVar[CacheManager[bytes] | None] = None
    _specialized_subclasses: ClassVar[list[type['Resource']]] = []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        has_discriminator = any((
            getattr(cls, 'storage_aliases', ()),
            getattr(cls, 'mimetypes', ()),
            getattr(cls, 'mimetype_prefixes', ()),
            getattr(cls, 'filename_suffixes', ()),
        ))
        if not has_discriminator:
            return

        Resource._specialized_subclasses = [
            subclass
            for subclass in Resource._specialized_subclasses
            if subclass is not cls
        ]
        Resource._specialized_subclasses.append(cls)

    def __new__(cls, *args, **kwargs):
        if cls is Resource and ('local_filepath' in kwargs or 'raw_data' in kwargs):
            return object.__new__(LocalResource)
        if cls is Resource:
            specialized_cls = cls._infer_specialized_resource_class(**kwargs)
            if specialized_cls is not Resource:
                return object.__new__(specialized_cls)
        return object.__new__(cls)

    @staticmethod
    def _normalize_token(value: str | None) -> str:
        return value.casefold() if isinstance(value, str) else ''

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _iter_specialized_subclasses(cls) -> list[type['Resource']]:
        _ensure_specialized_resource_types_loaded()
        return sorted(
            cls._specialized_subclasses,
            key=lambda subclass: subclass.resource_priority,
            reverse=True,
        )

    @classmethod
    def matches_payload(
        cls,
        *,
        storage: str | None = None,
        mimetype: str | None = None,
        filename: str | None = None,
    ) -> bool:
        storage_norm = cls._normalize_token(storage)
        mimetype_norm = cls._normalize_token(mimetype)
        filename_norm = cls._normalize_token(filename)

        if storage_norm and storage_norm in {value.casefold() for value in cls.storage_aliases}:
            return True
        if mimetype_norm and mimetype_norm in {value.casefold() for value in cls.mimetypes}:
            return True
        if mimetype_norm and any(
            mimetype_norm.startswith(prefix.casefold())
            for prefix in cls.mimetype_prefixes
        ):
            return True
        if filename_norm and any(
            filename_norm.endswith(suffix.casefold())
            for suffix in cls.filename_suffixes
        ):
            return True

        return False

    @classmethod
    def _infer_specialized_resource_class(cls, **kwargs) -> type['Resource']:
        storage = kwargs.get('storage')
        mimetype = kwargs.get('mimetype')
        filename = kwargs.get('filename')

        for subclass in cls._iter_specialized_subclasses():
            if subclass.matches_payload(
                storage=storage,
                mimetype=mimetype,
                filename=filename,
            ):
                return subclass

        return cls

    def _resolved_resource_class(self) -> type['Resource']:
        return type(self)._infer_specialized_resource_class(
            storage=getattr(self, 'storage', None),
            mimetype=getattr(self, 'mimetype', None),
            filename=getattr(self, 'filename', None),
        )

    @property
    def kind(self) -> str:
        return self._resolved_resource_class().resource_kind

    def _metadata_value(self, *keys: str) -> Any:
        value: Any = self.metadata
        for key in keys:
            if not isinstance(value, dict):
                return None
            value = value.get(key)
        return value

    @property
    def _cache(self) -> CacheManager[bytes]:
        if Resource._shared_cache is None:
            Resource._shared_cache = CacheManager[bytes]('resources',
                                                         enable_memory_cache=True,
                                                         memory_cache_maxsize=2)
        return Resource._shared_cache

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
        """Get the file data for this resource.

        This method automatically caches the file data locally. On subsequent
        calls, it checks the server for changes and uses cached data if unchanged.

        Args:
            use_cache: Cache behavior for this call. Use ``False`` to bypass
                cache entirely, ``True`` to read from and save to cache, or
                ``"loadonly"`` to read from cache without saving cache misses.
            auto_convert: If True, automatically converts to appropriate format (pydicom.Dataset, PIL Image, etc.)
            save_path: Optional path to save the file locally. If
                      ``use_cache=True``, the file is saved to save_path and
                      cache metadata points to that location (no duplication -
                      only one file on disk).

        Returns:
            File data (format depends on auto_convert and file type)

        Example:
            >>> resource = api.resources.get_list(project_name="My Project")[0]
            >>> data = resource.fetch_file_data(use_cache=True)
            >>> data = resource.fetch_file_data(use_cache="loadonly")
            >>> resource.fetch_file_data(save_path="local_copy")
        """
        # Version info for cache validation
        version_info = self._generate_version_info()

        # Download callback for the shared caching logic
        def download_callback(path: str | None) -> bytes:
            return self._api.download_resource_file(
                self,
                save_path=path,
                auto_convert=False
            )

        # Use shared caching logic from BaseEntity
        img_data = self._fetch_and_cache_file_data(
            cache_manager=self._cache,
            data_key=_IMAGE_CACHEKEY,
            version_info=version_info,
            download_callback=download_callback,
            save_path=save_path,
            use_cache=use_cache,
        )

        # # Persist discovery metadata alongside the cache entry (no-op when not cached)
        # if use_cache:
        #     self._cache.save_extra_info(self.id, {
        #         'upload_channel': self.upload_channel,
        #         'tags': self.tags or [],
        #     })

        if auto_convert:
            try:
                mimetype, _ = BaseApi._determine_mimetype(img_data, self.mimetype)
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
        """Get annotations associated with this resource.

        Example:
            >>> resource = api.resources.get_list(project_name="My Project")[0]
            >>> annotations = resource.fetch_annotations(annotation_type="segmentation")
            >>> [annotation.name for annotation in annotations]
        """

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

    def is_image(self) -> bool:
        """Check if the resource is a single-frame image."""
        return self.kind == 'image'

    def is_dicom(self) -> bool:
        """Check if the resource is a DICOM file.

        Returns:
            True if the resource is a DICOM file, False otherwise
        """
        return self.kind == 'dicom'

    def is_nifti(self) -> bool:
        """Check if the resource is a NIfTI file.

        Returns:
            True if the resource is a NIfTI file, False otherwise
        """
        return self.kind == 'nifti'

    def is_video(self) -> bool:
        """Check if the resource is a video file.

        Returns:
            True if the resource is a video file, False otherwise
        """
        return self.kind == 'video'

    def is_volume(self) -> bool:
        """Check if the resource is a volumetric resource."""
        return self.kind in {'volume', 'dicom', 'nifti'}

    def is_multiframe(self) -> bool:
        """Check if the resource contains multiple frames or slices."""
        return self.is_volume() or self.is_video()

    def get_depth(self) -> int:
        if self.is_image():
            return 1

        metadata = self.metadata

        if self.is_volume():
            frame_count = self._coerce_int(metadata.get('frame_count'))
            if frame_count is not None:
                return frame_count

        if self.is_video():
            streams = metadata.get('streams')
            if isinstance(streams, list):
                for stream in streams:
                    if not isinstance(stream, dict):
                        continue
                    if stream.get('codec_type') != 'video':
                        continue
                    frame_count = self._coerce_int(stream.get('nb_frames'))
                    if frame_count is not None:
                        return frame_count

        raise ValueError(f"Cannot determine depth for resource with mimetype {self.mimetype}")

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
        return f"{self.__class__.__name__}(id='{self.id}', filename='{self.filename}', size={self.size_mb}MB)"

    def __repr__(self) -> str:
        """Detailed string representation of the resource.

        Returns:
            Detailed string representation for debugging
        """
        return (
            f"{self.__class__.__name__}(id='{self.id}', filename='{self.filename}', "
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

    @property
    def _slice_cache_manager(self) -> CacheManager:
        """Cache manager for sliced volumes derived from this resource."""
        if not hasattr(self, '__slice_cache_manager'):
            self.__slice_cache_manager = CacheManager(
                'sliced_volumes',
                enable_memory_cache=True,
                memory_cache_maxsize=1,
            )
        return self.__slice_cache_manager

    @property
    def _frame_cache_manager(self) -> CacheManager:
        """Cache manager for sliced video frames derived from this resource."""
        if not hasattr(self, '__frame_cache_manager'):
            self.__frame_cache_manager = CacheManager(
                'sliced_video_frames',
                enable_memory_cache=True,
                memory_cache_maxsize=2,
            )
        return self.__frame_cache_manager

    def get_slice_resource(self, axis: 'ViewPlane', index: int) -> 'SlicedVolumeResource':
        """Get a proxy object for a specific volume slice."""
        if not self.is_volume():
            raise ValueError("Slices are only available for volume resources.")
        if index < 0:
            raise IndexError("slice index must be non-negative")

        from .sliced_resource import SlicedVolumeResource

        return SlicedVolumeResource(
            self,
            index,
            slice_axis=axis,
            sliced_vols_cache=self._slice_cache_manager,
        )

    def get_slice(self, axis: 'ViewPlane', index: int) -> 'np.ndarray':
        """Get a specific slice of the volume as a SlicedVolumeResource.

        Args:
            axis: The anatomical plane to slice along (e.g., 'axial', 'coronal', 'sagittal')
            index: The index of the slice along the specified axis
        Returns:
            A numpy array representing the specified slice
        """
        return self.get_slice_resource(axis, index).fetch_slice_data()

    def iter_slices(self, axis: 'ViewPlane') -> list['SlicedVolumeResource']:
        """Expand a volume into one proxy resource per slice."""
        if not self.is_volume():
            raise ValueError("Slices are only available for volume resources.")

        from .sliced_resource import SlicedVolumeResource

        return SlicedVolumeResource.slice_over(self, axis, self._slice_cache_manager)

    def get_frame_resource(self, index: int) -> 'SlicedVideoResource':
        """Get a proxy object for a specific video frame."""
        if not self.is_video():
            raise ValueError("Frames are only available for video resources.")
        if index < 0:
            raise IndexError("frame index must be non-negative")
        if index >= self.get_depth():
            raise IndexError(f"frame index {index} is out of bounds for resource depth {self.get_depth()}")

        from .sliced_video_resource import SlicedVideoResource

        return SlicedVideoResource(self, index, self._frame_cache_manager)

    def get_frame(self, index: int) -> 'np.ndarray':
        """Get a decoded video frame as a normalized array."""
        return self.get_frame_resource(index).fetch_frame_data()

    def iter_frames(self) -> list['SlicedVideoResource']:
        """Expand a video into one proxy resource per frame."""
        if not self.is_video():
            raise ValueError("Frames are only available for video resources.")

        from .sliced_video_resource import SlicedVideoResource

        return SlicedVideoResource.slice_over(self, self._frame_cache_manager)


class LocalResource(Resource):
    """Represents a local resource that hasn't been uploaded to DataMint API yet."""

    local_filepath: str | None = None
    raw_data: bytes | None = None

    @property
    def filepath_cached(self) -> Path | None:
        """Get the file path of the local resource data.

        Returns:
            Path to the local file, or None if only raw data is available.
        """
        if self.local_filepath is None:
            return None
        return Path(self.local_filepath)

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
                assert raw_data is not None
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
                local_data = {
                    **new_kwargs,
                    'local_filepath': None,
                    'raw_data': raw_data,
                }
                super().__init__(**local_data)
                return

        if convert_to_bytes and local_filepath:
            with open(local_filepath, 'rb') as f:
                raw_data = f.read()
                local_filepath = None
        if raw_data is not None:
            # import io
            assert raw_data is not None
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
            local_data = {
                **new_kwargs,
                'local_filepath': None,
                'raw_data': raw_data,
            }
            super().__init__(**local_data)
        elif local_filepath is not None:
            file_path = Path(local_filepath)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")

            mimetype, _ = guess_type(file_path)
            if mimetype is None or mimetype == DEFAULT_MIME_TYPE:
                logger.warning(f"Could not determine mimetype for file: {file_path}")
            size = file_path.stat().st_size
            created_at = datetime.fromtimestamp(file_path.stat().st_ctime).isoformat()

            local_data = {
                'id': "",
                'resource_uri': "",
                'storage': "",
                'location': str(file_path),
                'upload_channel': "",
                'filename': file_path.name,
                'modality': detect_modality(file_path),
                'mimetype': mimetype,
                'size': size,
                'upload_mechanism': "",
                'customer_id': "",
                'status': "local",
                'created_at': created_at,
                'created_by': "",
                'published': False,
                'deleted': False,
                'source_filepath': str(file_path),
                'local_filepath': str(file_path),
                'raw_data': None,
            }
            super().__init__(**local_data)

    @overload
    def fetch_file_data(
        self,
        *args,
        auto_convert: Literal[True] = True,
        save_path: str | None = None,
        use_cache: CacheMode = False,
        **kwargs,
    ) -> 'ImagingData': ...

    @overload
    def fetch_file_data(
        self,
        *args,
        auto_convert: Literal[False],
        save_path: str | None = None,
        use_cache: CacheMode = False,
        **kwargs,
    ) -> bytes: ...

    def fetch_file_data(
        self, *args,
        auto_convert: bool = True,
        save_path: str | None = None,
        use_cache: CacheMode = False,
        **kwargs,
    ) -> 'bytes | ImagingData':
        """Get the file data for this local resource.

        Args:
            auto_convert: If True, automatically converts to appropriate format (pydicom.Dataset, PIL Image, etc.)
            save_path: Optional path to save the file locally
            use_cache: Ignored for local resources; included for API parity.
        Returns:
            File data (format depends on auto_convert and file type)
        """
        self._resolve_cache_mode(use_cache)

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
                logger.error(f"Failed to auto-convert local resource {self}: {e}")
                raise

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


def _ensure_specialized_resource_types_loaded() -> None:
    global _SPECIALIZED_RESOURCE_TYPES_IMPORTED
    if _SPECIALIZED_RESOURCE_TYPES_IMPORTED:
        return

    from . import resources as resource_types

    _SPECIALIZED_RESOURCE_TYPES_IMPORTED = resource_types is not None
