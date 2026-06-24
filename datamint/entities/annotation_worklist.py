import logging
from typing import Any, cast

from pydantic import BaseModel, Field

from datamint.entities.annotations.annotation_spec import AnnotationSpec

from .base_entity import BaseEntity, MISSING_FIELD

_LOGGER = logging.getLogger(__name__)


class _WorklistAnnotatorInfo(BaseModel):
    email: str
    status: str
    expertise_level: str | None = None


class AnnotationWorklist(BaseEntity):

    def __init__(self, **kwargs):
        _LOGGER.info(f"Creating AnnotationWorklist with kwargs: {kwargs}")
        super().__init__(**kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AnnotationWorklist):
            return NotImplemented
        return self._comparison_payload() == other._comparison_payload()

    def _comparison_payload(self) -> dict[str, Any]:
        payload = self.asdict()

        if hasattr(self, 'annotators'):
            payload['annotators'] = sorted(
                [annotator.model_dump(warnings='none') for annotator in self.annotators],
                key=lambda annotator: (
                    annotator['email'],
                    annotator['status'],
                    annotator['expertise_level'] or '',
                ),
            )

        if hasattr(self, 'annotations'):
            payload['annotations'] = sorted(
                [_annotation_spec_payload(annotation) for annotation in self.annotations],
                key=lambda annotation: (
                    annotation.get('scope', ''),
                    annotation.get('identifier', ''),
                    str(annotation.get('type', '')),
                ),
            )

        if hasattr(self, 'resource_ids'):
            payload['resource_ids'] = sorted(self.resource_ids)

        return payload

    name: str
    status: str = 'new'
    annotators: list[_WorklistAnnotatorInfo]
    description: str | None = None
    annotations: list[AnnotationSpec] = Field(default=cast(list[AnnotationSpec], MISSING_FIELD))
    created_by: str = Field(default=MISSING_FIELD)
    created_at: str = Field(default=MISSING_FIELD)
    resource_ids: list[str] = Field(default=cast(list[str], MISSING_FIELD))


def _annotation_spec_payload(annotation: AnnotationSpec) -> dict[str, Any]:
    payload = annotation.asdict()
    values = payload.get('values')
    if isinstance(values, list):
        payload['values'] = sorted(values)
    return payload
