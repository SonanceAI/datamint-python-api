from __future__ import annotations

from typing import Any

from .annotation import Annotation


class ImageClassification(Annotation):
    def __init__(
        self,
        name: str | None = None,
        value: str | None = None,
        confiability: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if name is not None:
            kwargs.setdefault('identifier', name)
        if value is not None:
            kwargs.setdefault('text_value', value)

        kwargs.setdefault('confiability', confiability)
        kwargs.setdefault('scope', 'image')
        kwargs.setdefault('annotation_type', 'category')
        super().__init__(**kwargs)
