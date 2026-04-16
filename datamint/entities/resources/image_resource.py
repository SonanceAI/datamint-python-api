from __future__ import annotations

from typing import ClassVar

from ..resource import Resource


class ImageResource(Resource):
    """Represents a single-frame 2D image resource."""

    resource_kind: ClassVar[str] = 'image'
    resource_priority: ClassVar[int] = 10
    storage_aliases: ClassVar[tuple[str, ...]] = ('ImageResource', 'ImageResourceHandler')
    mimetype_prefixes: ClassVar[tuple[str, ...]] = ('image/',)

    @property
    def width(self) -> int | None:
        return self._coerce_int(self._metadata_value('width'))

    @property
    def height(self) -> int | None:
        return self._coerce_int(self._metadata_value('height'))

    def get_dimensions(self) -> tuple[int | None, int | None]:
        """Get image dimensions as ``(width, height)``."""
        return self.width, self.height

    def get_depth(self) -> int:
        return 1