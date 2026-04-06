from datamint.api.base_api import BaseApi
from typing import TYPE_CHECKING, Any
from datamint.entities import AnnotationSpec
from collections.abc import Sequence

if TYPE_CHECKING:
    from datamint.entities import Project


class AnnotationSetsApi(BaseApi):
    ENDPOINT_BASE = "/annotationsets"

    def get_segmentation_group(self, annotation_set: 'str | Project') -> dict:
        """Get the segmentation group for a given annotation set ID or Project."""

        if isinstance(annotation_set, str):
            annotation_set_id = annotation_set
        else:
            annotation_set_id = annotation_set.worklist_id

        endpoint = f"/{self.ENDPOINT_BASE}/{annotation_set_id}/segmentation-group"
        return self._make_request("GET", endpoint).json()

    def get_annotations_specs(self, annotation_set: 'str | Project') -> Sequence[AnnotationSpec]:
        """Get the annotations specs for a given annotation set ID or Project."""

        if isinstance(annotation_set, str):
            annotation_set_id = annotation_set
        else:
            annotation_set_id = annotation_set.worklist_id

        result = self._get_by_id(annotation_set_id)

        return [AnnotationSpec(**annspec) for annspec in result['annotations']]

    def _get_by_id(self, annotation_set_id: str) -> dict[str, Any]:
        """Get an annotation set by its ID.

        Args:
            annotation_set_id: The ID of the annotation set to retrieve.

        Returns:
            A dictionary representing the annotation set.
        """
        endpoint = f"/{self.ENDPOINT_BASE}/{annotation_set_id}"
        result = self._make_request("GET", endpoint).json()

        result['annotations'] = [AnnotationSpec(**annspec) for annspec in result['annotations']]
        return result
