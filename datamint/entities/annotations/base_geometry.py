from __future__ import annotations

from typing import Any, TypeVar

from .annotation import Annotation
from .geometry import Geometry

GeometryT = TypeVar('GeometryT', bound=Geometry)


class BaseGeometryAnnotation(Annotation):
    """Base entity for typed geometry annotations."""

    geometry: Geometry | None = None

    def __init__(self, geometry: Geometry | dict[str, Any] | None = None, **kwargs: Any) -> None:
        kwargs.setdefault('scope', 'frame' if kwargs.get('frame_index') is not None else 'image')
        super().__init__(geometry=geometry, **kwargs)

    @staticmethod
    def _coerce_geometry(
        value: GeometryT | dict[str, Any] | list[Any] | tuple[Any, ...] | None,
        geometry_cls: type[GeometryT],
    ) -> GeometryT | None:
        if value is None:
            return None

        if isinstance(value, geometry_cls):
            return value

        if isinstance(value, dict):
            return geometry_cls(**value)

        if isinstance(value, (list, tuple)):
            return geometry_cls(points=value)

        raise TypeError(f'Unsupported geometry payload type: {type(value)}')