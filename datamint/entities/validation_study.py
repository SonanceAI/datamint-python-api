from typing import Any

from .base_entity import BaseEntity


class ValidationStudy(BaseEntity):
    """A validation study: readers evaluate a model's predictions. """

    name: str
    model_name: str
    model_version: int | None = None
    description: str | None = None
    project_id: str | None = None
    resource_ids: list[str] | None = None
    reader_tasks: dict[str, Any] | None = None
    status: str | None = None
