import logging
import sys
from typing import Any, TYPE_CHECKING
from pydantic import ConfigDict, BaseModel, PrivateAttr

if TYPE_CHECKING:
    from datamint.api.client import Api
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
            if hasattr(self, field_name) and getattr(self, field_name) == MISSING_FIELD:
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
