from __future__ import annotations

import json
from typing import Any, TypeVar

from .annotation import Annotation
from .geometry import Geometry

GeometryT = TypeVar('GeometryT', bound=Geometry)


class BaseGeometryAnnotation(Annotation):
    """Base entity for typed geometry annotations."""

    geometry: Geometry | None = None

    payload: dict[str, Any] | None = None
    """Optional free-form payload stored as the annotation ``value``."""

    def __init__(self, geometry: Geometry | dict[str, Any] | None = None, **kwargs: Any) -> None:
        if 'scope' not in kwargs:
            inferred_scope = 'image'
            if isinstance(geometry, dict):
                coord_system = geometry.get('coordinate_system', 'patient')
            else:
                coord_system = getattr(geometry, 'coordinate_system', 'patient')
            if 'frame_index' in kwargs and coord_system != 'patient':
                inferred_scope = 'frame'
            kwargs['scope'] = inferred_scope

        super().__init__(geometry=geometry, **kwargs)

    def _to_create_dto(self):
        """Convert this annotation entity into a create DTO.

        Returns:
            The create DTO produced by the base class, with ``value`` replaced
            by the JSON-serialized :attr:`payload` when a payload is set.
        """
        dto = super()._to_create_dto()
        if self.payload is not None:
            dto.value = json.dumps(self.payload)
        return dto

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
