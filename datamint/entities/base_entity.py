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
        return self.model_dump()

    def asjson(self) -> str:
        """Convert the entity to a JSON string, including unknown fields."""
        return self.model_dump_json()

    def model_post_init(self, __context: Any) -> None:
        if self.__pydantic_extra__:
            _LOGGER.warning(f"Unknown fields found in {self.__class__.__name__} "
                            f"fields: {self.__pydantic_extra__.keys()}. ")
