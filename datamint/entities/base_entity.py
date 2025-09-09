import logging
import sys
from typing import Any
from pydantic import ConfigDict, BaseModel

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
    """

    model_config = ConfigDict(extra='allow')  # Allow extra fields not defined in the model

    def asdict(self) -> dict[str, Any]:
        """Convert the entity to a dictionary, including unknown fields."""
        return self.model_dump(warnings='none')

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
