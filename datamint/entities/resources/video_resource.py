from __future__ import annotations

from typing import Any, ClassVar, TYPE_CHECKING

from ..resource import Resource

if TYPE_CHECKING:
    from ..sliced_video_resource import SlicedVideoResource


class VideoResource(Resource):
    """Represents a video resource with per-frame access helpers."""

    resource_kind: ClassVar[str] = 'video'
    resource_priority: ClassVar[int] = 20
    storage_aliases: ClassVar[tuple[str, ...]] = ('VideoResource', 'VideoResourceHandler')
    mimetype_prefixes: ClassVar[tuple[str, ...]] = ('video/',)

    def _video_stream_metadata(self) -> dict[str, Any] | None:
        streams = self._metadata_value('streams')
        if not isinstance(streams, list):
            return None

        for stream in streams:
            if isinstance(stream, dict) and stream.get('codec_type') == 'video':
                return stream

        return None

    @property
    def frame_count(self) -> int:
        return self.get_depth()

    @property
    def width(self) -> int | None:
        stream = self._video_stream_metadata()
        if stream is None:
            return None
        return self._coerce_int(stream.get('width'))

    @property
    def height(self) -> int | None:
        stream = self._video_stream_metadata()
        if stream is None:
            return None
        return self._coerce_int(stream.get('height'))

    def get_dimensions(self) -> tuple[int | None, int | None]:
        """Get video frame dimensions as ``(width, height)``."""
        return self.width, self.height

    def get_depth(self) -> int:
        frame_count = self._coerce_int(self._metadata_value('frame_count'))
        if frame_count is not None:
            return frame_count

        stream = self._video_stream_metadata()
        if stream is None:
            raise ValueError(f"Cannot determine frame count for video resource {self.filename!r}")

        frame_count = self._coerce_int(stream.get('nb_frames'))
        if frame_count is None:
            raise ValueError(f"Cannot determine frame count for video resource {self.filename!r}")

        return frame_count

    def iter_frames(self) -> list['SlicedVideoResource']:
        return super().iter_frames()