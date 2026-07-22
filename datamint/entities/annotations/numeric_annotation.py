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

        kwargs.setdefault('confiability', confiability)
        kwargs.setdefault('scope', 'image')
        super().__init__(**kwargs)
