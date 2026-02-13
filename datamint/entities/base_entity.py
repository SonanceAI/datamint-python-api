import logging
import sys
from typing import Any, TYPE_CHECKING
from pydantic import ConfigDict, BaseModel, PrivateAttr

if TYPE_CHECKING:
    from datamint.api.entity_base_api import EntityBaseApi

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self
_LOGGER = logging.getLogger(__name__)

MISSING_FIELD = 'MISSING_FIELD'  # Used when a field is sometimes missing for one endpoint but not on another endpoint

# Track logged warnings to avoid duplicates
_LOGGED_WARNINGS: set[tuple[str, str]] = set()


class BaseEntity(BaseModel):
    """
    Base class for all entities in the Datamint system.

    This class provides common functionality for all entities, such as
    serialization and deserialization from dictionaries, as well as
    handling unknown fields gracefully.

    The API client is automatically injected by the Api class when entities
    are created through API endpoints.
    """

    model_config = ConfigDict(extra='allow',
                              arbitrary_types_allowed=True,  # Allow extra fields and arbitrary types
                              ser_json_bytes='base64',
                              val_json_bytes='base64')

    _api: 'EntityBaseApi[Self] | EntityBaseApi' = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        # check attributes for MISSING_FIELD and delete them
        for field_name in self.__pydantic_fields__.keys():
            if hasattr(self, field_name) and type(getattr(self, field_name)) == str and getattr(self, field_name) == MISSING_FIELD:
                delattr(self, field_name)

    def asdict(self) -> dict[str, Any]:
        """Convert the entity to a dictionary, including unknown fields."""
        d = self.model_dump(warnings='none')
        return {k: v for k, v in d.items() if v != MISSING_FIELD}

    def asjson(self) -> str:
        """Convert the entity to a JSON string, including unknown fields."""
        return self.model_dump_json(warnings='none')

    def model_post_init(self, __context: Any) -> None:
        """Handle unknown fields by logging a warning once per class/field combination in debug mode."""
        if self.__pydantic_extra__ and _LOGGER.isEnabledFor(logging.DEBUG):
            class_name = self.__class__.__name__

            have_to_log = False
            for key in self.__pydantic_extra__.keys():
                warning_key = (class_name, key)

                if warning_key not in _LOGGED_WARNINGS:
                    _LOGGED_WARNINGS.add(warning_key)
                    have_to_log = True

            if have_to_log:
                _LOGGER.warning(f"Unknown fields {list(self.__pydantic_extra__.keys())} found in {class_name}")

    def is_attr_missing(self, attr_name: str) -> bool:
        """Check if a value is the MISSING_FIELD sentinel."""
        if attr_name not in self.__pydantic_fields__.keys():
            raise AttributeError(f"Attribute '{attr_name}' not found in entity of type '{self.__class__.__name__}'")
        if not hasattr(self, attr_name):
            return True
        return getattr(self, attr_name) == MISSING_FIELD  # deprecated

    def _refresh(self) -> Self:
        """Refresh the entity data from the server.

        This method fetches the latest data from the server and updates
        the current instance with any missing or updated fields.

        Returns:
            The updated Entity instance (self)
        """
        updated_ent = self._api.get_by_id(self._api._entid(self))

        # Update all fields from the fresh data
        for field_name, field_value in updated_ent.model_dump().items():
            if field_value != MISSING_FIELD:
                setattr(self, field_name, field_value)

        return self

    def _ensure_attr(self, attr_name: str) -> None:
        """Ensure that a given attribute is not MISSING_FIELD, refreshing if necessary.

        Args:
            attr_name: Name of the attribute to check and ensure
        """
        if attr_name not in self.__pydantic_fields__.keys():
            raise AttributeError(f"Attribute '{attr_name}' not found in entity of type '{self.__class__.__name__}'")

        if self.is_attr_missing(attr_name):
            self._refresh()

    def has_missing_attrs(self) -> bool:
        """Check if the entity has any attributes that are MISSING_FIELD.

        Returns:
            True if any attribute is MISSING_FIELD, False otherwise
        """
        return any(self.is_attr_missing(attr_name) for attr_name in self.__pydantic_fields__.keys())

    def _fetch_and_cache_file_data(
        self,
        cache_manager: 'Any',  # CacheManager[bytes]
        data_key: str,
        version_info: dict[str, Any],
        download_callback: 'Any',  # Callable[[str | None], bytes]
        save_path: str | None = None,
        use_cache: bool = False,
    ) -> bytes:
        """Shared logic for fetching and caching file data.

        This method handles the caching strategy for both Resource and Annotation entities.

        Args:
            cache_manager: The CacheManager instance to use
            data_key: Key identifying the type of data (e.g., 'image_data', 'annotation_data')
            version_info: Version information for cache validation
            download_callback: Function to call to download the file, takes save_path as parameter
            save_path: Optional path to save the file locally
            use_cache: If True, uses cached data when available

        Returns:
            File data as bytes
        """
        from pathlib import Path

        # Try to get from cache
        img_data = None
        
        if use_cache:
            img_data = cache_manager.get(self.id, data_key, version_info)
            if img_data is not None:
                _LOGGER.debug(f"Using cached data for {self.__class__.__name__} {self.id}")

        if img_data is None:
            # Cache miss - fetch from server
            if use_cache and save_path:
                # Download directly to save_path, register location in cache metadata
                _LOGGER.debug(f"Downloading to save_path: {save_path}")
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                
                img_data = download_callback(save_path)
                
                # Register save_path in cache metadata (no file duplication)
                cache_manager.register_file_location(
                    self.id, data_key, save_path, version_info
                )
            elif use_cache:
                # No save_path - download to cache directory
                cache_path = cache_manager.get_expected_path(self.id, data_key)
                _LOGGER.debug(f"Downloading to cache: {cache_path}")
                
                img_data = download_callback(str(cache_path))
                
                # Register in cache metadata
                cache_manager.set(self.id, data_key, img_data, version_info)
            else:
                # No caching - direct download to save_path (or just return bytes)
                _LOGGER.debug(f"Fetching data from server for {self.__class__.__name__} {self.id}")
                img_data = download_callback(save_path)
        elif save_path:
            # Cached data found, but user wants to save to a specific path
            _LOGGER.debug(f"Saving cached data to specified path: {save_path}")
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_data)

        return img_data
