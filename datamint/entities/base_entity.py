import logging
import sys
from html import escape
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, PrivateAttr

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

# ---------------------------------------------------------------------------
# Jinja2 HTML template for BaseEntity Jupyter repr
# ---------------------------------------------------------------------------
_ENTITY_HTML_TEMPLATE = """\
<div style="max-width: 720px; margin: 10px 0; overflow: hidden; border-radius: 18px;
           border: 1px solid var(--vscode-panel-border, #d0d7de);
           background: var(--vscode-editor-background, #ffffff);
           color: var(--vscode-foreground, #1f2328);
           box-shadow: 0 10px 30px rgba(15, 23, 42, 0.10);">

  {# ---- Header ---- #}
  <div style="padding: 18px 20px;
             border-bottom: 1px solid var(--vscode-panel-border, #d0d7de);
             background: linear-gradient(135deg, rgba(59, 130, 246, 0.14), rgba(16, 185, 129, 0.08));">
    <div style="font-size: 11px; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
               color: var(--vscode-descriptionForeground, #57606a);">Entity</div>
    <div style="display: flex; align-items: center; justify-content: space-between;
               gap: 12px; flex-wrap: wrap; margin-top: 8px;">
      <h4 style="margin: 0; font-size: 22px; font-weight: 700; color: inherit;">{{ entity_name }}</h4>
    </div>
  </div>

  {# ---- Fields table ---- #}
  {%- if fields %}
  <div style="padding: 12px 20px 18px;">
    <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
      {%- for name, value in fields %}
      <tr>
        <th style="padding: 10px 12px 10px 0; width: 30%; text-align: left; vertical-align: top;
                  font-size: 11px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
                  color: var(--vscode-descriptionForeground, #57606a); white-space: nowrap;">{{ name }}</th>
        <td style="padding: 10px 0; border-bottom: 1px solid var(--vscode-panel-border, #d0d7de);">
          <span style="display: inline-block; padding: 2px 8px; border-radius: 999px;
                      background: var(--vscode-textCodeBlock-background, #f6f8fa);
                      color: var(--vscode-textPreformat-foreground, var(--vscode-foreground, #1f2328));
                      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                      font-size: 13px;"
          >{{ value }}</span>
        </td>
      </tr>
      {%- endfor %}
    </table>
  </div>
  {%- else %}
  <div style="padding: 18px 20px; font-size: 14px;
             color: var(--vscode-descriptionForeground, #57606a);">No non-empty fields to display.</div>
  {%- endif %}

</div>
"""

_entity_template = None


def _get_entity_template():
    """Lazily compile and cache the Jinja2 entity template."""
    global _entity_template
    if _entity_template is None:
        from jinja2 import Environment
        _entity_template = Environment(autoescape=True).from_string(_ENTITY_HTML_TEMPLATE)
    return _entity_template


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
            if value is None or value == '':
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
        entity_id = getattr(self, 'id', None)
        fields = self._get_display_fields()

        return _get_entity_template().render(
            entity_name=self.__class__.__name__,
            entity_id=str(entity_id) if entity_id else None,
            fields=fields,
        )

    def __str__(self) -> str:
        fields = self._get_display_fields()

        header = self.__class__.__name__

        if not fields:
            return f"{header}\n  (no non-empty fields)"

        lines = [header] + [f"  {name}: {value}" for name, value in fields]
        return "\n".join(lines)

    def __init__(self, **data):
        super().__init__(**data)
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

    def has_missing_attrs(self) -> bool:
        """Check if the entity has any attributes that are MISSING_FIELD.

        Returns:
            True if any attribute is MISSING_FIELD, False otherwise
        """
        return any(self.is_attr_missing(attr_name) for attr_name in self.__pydantic_fields__.keys())


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
            if img_data is not None:
                _LOGGER.debug("Using cached data for %s %s", self.__class__.__name__, self.id)

        if img_data is None:
            # Cache miss - fetch from server
            if should_save_to_cache and save_path:
                # Download directly to save_path, register location in cache metadata
                _LOGGER.debug("Downloading to save_path: %s", save_path)
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
