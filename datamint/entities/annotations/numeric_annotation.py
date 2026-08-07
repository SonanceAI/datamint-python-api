from __future__ import annotations

from typing import Any

from .annotation import Annotation
from .types import AnnotationType


class NumericAnnotation(Annotation):
    def __init__(
        self,
        name: str | None = None,
        value: int | float | None = None,
        units: str | None = None,
        confiability: float = 1.0,
        worklist_id: str | None = None,
        author_email: str | None = None,
        imported_from: str | None = None,
        model_id: str | None = None,
        source: str | None = None,
        **kwargs: Any,
    ) -> None:
        if name is not None:
            kwargs.setdefault('identifier', name)
        if value is not None:
            kwargs.setdefault('numeric_value', value)
            is_int = isinstance(value, int) and not isinstance(value, bool)
            kwargs.setdefault('annotation_type', AnnotationType.INTEGER if is_int else AnnotationType.FLOAT)

        if units is not None:
            kwargs.setdefault('units', units)
        if worklist_id is not None:
            kwargs.setdefault('annotation_worklist_id', worklist_id)
        if author_email is not None:
            kwargs.setdefault('import_author', author_email)
        if imported_from is not None:
            kwargs.setdefault('imported_from', imported_from)
        if model_id is not None:
            kwargs.setdefault('model_id', model_id)
        if source is not None:
            kwargs.setdefault('source', source)

        kwargs.setdefault('confiability', confiability)
        kwargs.setdefault('scope', 'image')
        super().__init__(**kwargs)
