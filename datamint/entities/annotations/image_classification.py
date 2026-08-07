from __future__ import annotations

from typing import Any

from .annotation import Annotation


class ImageClassification(Annotation):
    def __init__(
        self,
        name: str | None = None,
        value: str | None = None,
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
            kwargs.setdefault('text_value', value)
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
        kwargs.setdefault('annotation_type', 'category')
        super().__init__(**kwargs)
