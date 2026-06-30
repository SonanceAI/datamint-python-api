from __future__ import annotations
from typing import TYPE_CHECKING, Any
import logging
from functools import cached_property
from medimgkit.readers import read_array_normalized

if TYPE_CHECKING:
    from datamint.entities import Resource

_LOGGER = logging.getLogger(__name__)


class SlicedResourceBase:
    """Base class for sliced resources, which are proxies that present a subset of a resource's data.

    This class provides common functionality for sliced resources, such as caching and attribute access.
    """

    def __init__(self, parent: Resource):
        self._parent = parent

    def __getstate__(self):
        state = self.__dict__.copy()
        if '_api' in state:
            _LOGGER.info("Removing _api from SlicedResourceBase state for pickling."
                         " It shouldn't be there.")
            del state['_api']
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _get_version_info(self) -> dict:
        """Get version info from the parent resource for cache validation."""
        return {
            'created_at': getattr(self._parent, 'created_at', None),
            'deleted_at': getattr(self._parent, 'deleted_at', None),
            'size': getattr(self._parent, 'size', None),
        }

    @cached_property
    def data_metainfo(self) -> dict:
        """Volume metadata. Loaded once and cached for the lifetime of this resource."""
        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        _, metainfo = read_array_normalized(raw, return_metainfo=True)
        return metainfo

    @property
    def parent_resource(self) -> Resource:
        """The original video Resource being proxied."""
        return self._parent

    def is_cached(self) -> bool:
        return self._parent.is_cached()
