import logging
import sys
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, PrivateAttr

from datamint._repr_utils import render_html_card, render_text_block
from datamint.types import CacheMode

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


class BaseEntityModel(BaseModel):
    """Shared lightweight Pydantic base for Datamint entities and DTOs."""

    model_config = ConfigDict(
        extra='allow',
        arbitrary_types_allowed=True,
        ser_json_bytes='base64',
        val_json_bytes='base64',
    )

    id: str

    def _get_display_fields(self, max_value_len: int = 120) -> list[tuple[str, str]]:
        """Collect non-empty, non-default fields for display purposes."""
        json_schema = self.model_json_schema()
        required_fields: set[str] = set(json_schema.get('required', []))

        fields: list[tuple[str, str]] = []
        for name, field_info in json_schema.get('properties', {}).items():
            if name == 'id':
                continue
            if name.endswith('_id'):
                continue
            value = getattr(self, name, None)
            try:
                if value is None or value == '':
                    continue
            except ValueError:
                continue
            if name not in required_fields:
                default_value = field_info.get('default')
                if default_value == MISSING_FIELD:
                    continue
                if default_value is not None and value == default_value:
                    continue
            display_value = str(value)
            if len(display_value) > max_value_len:
                display_value = display_value[:max_value_len - 3] + '...'
            fields.append((name.replace('_', ' ').title(), display_value))

        return fields

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter Notebooks."""
        return render_html_card(kind='Entity', name=self.__class__.__name__, fields=self._get_display_fields())

    def __str__(self) -> str:
        return render_text_block(self.__class__.__name__, self._get_display_fields())

    def __init__(self, **data):
        super().__init__(**data)
        for field_name in self.__pydantic_fields__:
            if hasattr(self, field_name) and isinstance(getattr(self, field_name), str) and getattr(self, field_name) == MISSING_FIELD:
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
            for key in self.__pydantic_extra__:
                warning_key = (class_name, key)

                if warning_key not in _LOGGED_WARNINGS:
                    _LOGGED_WARNINGS.add(warning_key)
                    have_to_log = True

            if have_to_log:
                _LOGGER.warning(f"Unknown fields {list(self.__pydantic_extra__.keys())} found in {class_name}")


class BaseEntity(BaseEntityModel):
    """
    Base class for all entities in the Datamint system.

    This class provides common functionality for all entities, such as
    serialization and deserialization from dictionaries, as well as
    handling unknown fields gracefully.

    The API client is automatically injected by the Api class when entities
    are created through API endpoints.
    """

    _api: 'EntityBaseApi[Self] | EntityBaseApi' = PrivateAttr()

    def _refresh(self) -> Self:
        """Refresh the entity data from the server.

        This method fetches the latest data from the server and updates
        the current instance with any missing or updated fields.

        Returns:
            The updated Entity instance (self)
        """
        updated_ent = self._api.get_by_id(self._api._entid(self))

        # Update declared fields directly from the model instance so that
        # nested Pydantic models are preserved as model instances rather than
        # being converted to plain dicts (as model_dump() would do).
        for field_name in updated_ent.__pydantic_fields__:
            if updated_ent._raw_hasattr(field_name):
                setattr(self, field_name, updated_ent._getraw_attr(field_name))

        # Also propagate any extra (unknown) fields
        if updated_ent.__pydantic_extra__:
            for field_name, field_value in updated_ent.__pydantic_extra__.items():
                if field_value != MISSING_FIELD:
                    setattr(self, field_name, field_value)

        return self

    def _getraw_attr(self, name: str) -> Any:
        """Get the raw attribute value without triggering automatic refresh."""
        return object.__getattribute__(self, name)  # Pydantic has this implemented

    def _raw_hasattr(self, name: str) -> bool:
        try:
            v = self._getraw_attr(name)
            if v is MISSING_FIELD:
                return False
            return True
        except AttributeError:
            return False

    def __getattr__(self, name: str) -> Any:
        """Intercept access to missing fields and trigger automatic refresh.

        When a field has the MISSING_FIELD sentinel value, it is deleted in __init__.
        This method catches subsequent attribute access attempts and automatically
        refreshes the entity from the server to populate all missing fields.

        """
        # Delegate private attributes to Pydantic's handler (manages __pydantic_private__).
        # Defining our own __getattr__ overrides Pydantic's, so we must forward these.
        if name.startswith('_'):
            try:
                return super().__getattr__(name)  # pydantic implements __getattr__
            except AttributeError:
                try:
                    return object.__getattribute__(self, name)
                except AttributeError:
                    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # Check if this is a declared Pydantic field that was deleted (MISSING_FIELD)
        try:
            pydantic_fields = object.__getattribute__(self, '__pydantic_fields__')
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        if name not in pydantic_fields:
            # Not a declared field - fall back to pydantic's extra fields (extra='allow')
            pydantic_extra = object.__getattribute__(self, '__pydantic_extra__')
            if pydantic_extra is not None and name in pydantic_extra:
                return pydantic_extra[name]
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

        # This is a declared field that was deleted (MISSING_FIELD sentinel).
        # Try to refresh from the server using _api stored in __pydantic_private__.
        try:
            private_state = object.__getattribute__(self, '__pydantic_private__') or {}
            api_ref = private_state.get('_api')
            if api_ref is not None:
                _LOGGER.debug("Refreshing %s from server since %s is missing",
                              type(self).__name__, name)
                self._refresh()
                # After refresh, the field should be populated. If not, raise AttributeError
                if self._raw_hasattr(name):
                    return self._getraw_attr(name)
        except Exception as e:
            # If refresh fails for any reason, fall through to standard error
            raise

        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def _ensure_attr(self, attr_name: str) -> None:
        """Ensure that a given attribute is not MISSING_FIELD, refreshing if necessary.

        Args:
            attr_name: Name of the attribute to check and ensure
        """
        if attr_name not in self.__pydantic_fields__:
            raise AttributeError(f"Attribute '{attr_name}' not found in entity of type '{self.__class__.__name__}'")

        if self.is_attr_missing(attr_name):
            self._refresh()

    def __getstate__(self) -> dict:
        state = super().__getstate__()
        # Strip _api (contains unpicklable connections)
        if state.get('__pydantic_private__') is not None:
            state = dict(state)
            state['__pydantic_private__'] = {
                k: v for k, v in state['__pydantic_private__'].items() if k != '_api'
            }
        return state

    def __setstate__(self, state: dict) -> None:
        if state.get('__pydantic_private__') is not None:
            state = dict(state)
            private = dict(state['__pydantic_private__'])
            private['_api'] = None  # placeholder;
            state['__pydantic_private__'] = private
        super().__setstate__(state)

    @staticmethod
    def _resolve_cache_mode(use_cache: CacheMode) -> tuple[bool, bool]:
        if isinstance(use_cache, str):
            if use_cache != 'loadonly':
                raise ValueError("use_cache must be False, True, or 'loadonly'.")
            return True, False

        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be False, True, or 'loadonly'.")

        return use_cache, use_cache

    def _fetch_and_cache_file_data(
        self,
        cache_manager: 'Any',  # CacheManager[bytes]
        data_key: str,
        version_info: dict[str, Any],
        download_callback: 'Any',  # Callable[[str | None], bytes]
        save_path: str | None = None,
        use_cache: CacheMode = False,
    ) -> bytes:
        """Shared logic for fetching and caching file data.

        This method handles the caching strategy for both Resource and Annotation entities.

        Args:
            cache_manager: The CacheManager instance to use
            data_key: Key identifying the type of data (e.g., 'image_data', 'annotation_data')
            version_info: Version information for cache validation
            download_callback: Function to call to download the file, takes save_path as parameter
            save_path: Optional path to save the file locally
            use_cache: Cache behavior for this call. ``False`` disables cache,
                ``True`` enables cache reads and writes, and ``"loadonly"`` reads
                from cache without saving cache misses back.

        Returns:
            File data as bytes
        """
        from pathlib import Path

        # Try to get from cache
        img_data = None
        should_load_from_cache, should_save_to_cache = self._resolve_cache_mode(use_cache)

        if should_load_from_cache:
            img_data = cache_manager.get(self.id, data_key, version_info)

        if img_data is None:
            # Cache miss - fetch from server
            if should_save_to_cache and save_path:
                # Download directly to save_path, register location in cache metadata
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)

                img_data = download_callback(save_path)

                # Register save_path in cache metadata (no file duplication)
                cache_manager.register_file_location(
                    self.id, data_key, save_path, version_info
                )
            elif should_save_to_cache:
                # No save_path - download to cache directory
                cache_path = cache_manager.get_expected_path(self.id, data_key)
                _LOGGER.debug("Downloading to cache: %s", cache_path)

                img_data = download_callback(str(cache_path))

                # Register in cache metadata
                cache_manager.set(self.id, data_key, img_data, version_info)
            else:
                # No caching - direct download to save_path (or just return bytes)
                if should_load_from_cache and not should_save_to_cache:
                    _LOGGER.debug(
                        "Cache miss for %s %s; downloading without updating cache", self.__class__.__name__, self.id
                    )
                else:
                    _LOGGER.debug("Fetching data from server for %s %s", self.__class__.__name__, self.id)
                img_data = download_callback(save_path)
        elif save_path:
            # Cached data found, but user wants to save to a specific path
            _LOGGER.debug("Saving cached data to specified path: %s", save_path)
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                f.write(img_data)

        return img_data

    def is_attr_missing(self, attr_name: str) -> bool:
        """Check if a value is the MISSING_FIELD sentinel."""
        if attr_name not in self.__pydantic_fields__:
            raise AttributeError(f"Attribute '{attr_name}' not found in entity of type '{self.__class__.__name__}'")
        if not self._raw_hasattr(attr_name):
            return True
        return getattr(self, attr_name) == MISSING_FIELD

    def has_missing_attrs(self) -> bool:
        """Check if the entity has any attributes that are MISSING_FIELD.

        Returns:
            True if any attribute is MISSING_FIELD, False otherwise
        """
        return any(self.is_attr_missing(attr_name) for attr_name in self.__pydantic_fields__)
