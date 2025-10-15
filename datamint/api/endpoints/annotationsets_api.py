from datamint.api.base_api import BaseApi
import logging

_LOGGER = logging.getLogger(__name__)


class AnnotationSetsApi(BaseApi):
    def get_segmentation_group(self, annotation_set_id: str) -> dict:
        """Get the segmentation group for a given annotation set ID."""
        endpoint = f"/annotationsets/{annotation_set_id}/segmentation-group"
        return self._make_request("GET", endpoint).json()
