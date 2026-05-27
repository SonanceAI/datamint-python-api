from typing import Any

from pydantic import BaseModel, ConfigDict
from .types import AnnotationType


class AnnotationSpec(BaseModel):
    """Base class for annotation specifications.
    Used by the API to define the expected structure of annotations for a given project.
    """
    model_config = ConfigDict(extra='allow',
                              ser_json_bytes='base64',
                              val_json_bytes='base64')

    type: AnnotationType
    scope: str
    required: bool
    identifier: str

    def __new__(cls, *args, **kwargs):
        if cls is AnnotationSpec and kwargs.get('type') == AnnotationType.CATEGORY:
            return super().__new__(CategoryAnnotationSpec)  # type: ignore
        return super().__new__(cls)

    @classmethod
    def create(cls, **kwargs) -> 'AnnotationSpec':
        """Factory method to create the appropriate AnnotationSpec subclass based on type."""
        annotation_type = kwargs.get('type')

        if annotation_type == AnnotationType.CATEGORY:
            return CategoryAnnotationSpec(**kwargs)

        return cls(**kwargs)

    def asdict(self):
        """Convert the entity to a dictionary, including unknown fields."""
        return self.model_dump(warnings='none', exclude_none=True)

    def asjson(self) -> str:
        """Convert the entity to a JSON string, including unknown fields."""
        return self.model_dump_json(warnings='none', exclude_none=True)


class CategoryAnnotationSpec(AnnotationSpec):
    type: AnnotationType = AnnotationType.CATEGORY
    values: list[str]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CategoryAnnotationSpec):
            return NotImplemented
        return self._comparison_payload() == other._comparison_payload()

    def _comparison_payload(self) -> dict[str, Any]:
        payload = self.model_dump(warnings='none')
        payload['values'] = sorted(self.values)
        return payload
