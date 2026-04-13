from datamint.api.base_api import BaseApi
from typing import TYPE_CHECKING, Any
from datamint.entities import AnnotationSpec
from collections.abc import Sequence

if TYPE_CHECKING:
    from datamint.entities import Project


class AnnotationSetsApi(BaseApi):
    ENDPOINT_BASE = "/annotationsets"

    def create(self,
               name: str,
               resource_ids: list[str],
               description: str | None = None,
               annotations: list[dict] | None = None,
               annotators: list[str] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               ) -> str:
        """Create a new annotation worklist.

        Args:
            name: Name of the annotation worklist.
            resource_ids: List of resource IDs to include.
            description: Optional description.
            annotations: Optional list of annotation spec dicts.
            annotators: Optional list of annotator email addresses.
            frame_labels: Optional list of frame label names.
            image_labels: Optional list of image label names.

        Returns:
            The ID of the created annotation set.
        """
        payload: dict = {'name': name, 'resource_ids': resource_ids}
        if description is not None:
            payload['description'] = description
        if annotations is not None:
            payload['annotations'] = annotations
        if annotators is not None:
            payload['annotators'] = annotators
        if frame_labels is not None:
            payload['frame_labels'] = frame_labels
        if image_labels is not None:
            payload['image_labels'] = image_labels
        response = self._make_request('POST', f'{self.ENDPOINT_BASE}', json=payload)
        respdata = response.json()
        if isinstance(respdata, dict):
            return respdata.get('id')
        return respdata

    def update(self, annotation_set_id: str, **kwargs) -> None:
        """Partially update an annotation worklist.

        Args:
            annotation_set_id: The annotation set ID to update.
            **kwargs: Fields to update (e.g. ``name``, ``annotators``,
                ``resource_ids``, ``status``).
        """
        payload = {k: v for k, v in kwargs.items() if v is not None}
        self._make_request('PATCH', f'{self.ENDPOINT_BASE}/{annotation_set_id}', json=payload)

    def update_segmentation_group(self,
                                  annotation_set: 'str | Project',
                                  definitions: list[dict],
                                  segmentation_value_type: str = 'single_label',
                                  renames: list[str] | None = None) -> None:
        """Replace the segmentation-group definitions for an annotation set.

        Args:
            annotation_set: The annotation set ID or Project instance.
            definitions: List of definition dicts with keys ``identifier``,
                ``color``, and ``index``.
            segmentation_value_type: ``'single_label'`` (default) or
                ``'multi_label'``.
            renames: Optional rename pairs (old→new identifier strings).
        """
        if isinstance(annotation_set, str):
            annotation_set_id = annotation_set
        else:
            annotation_set_id = annotation_set.worklist_id

        payload: dict = {
            'segmentationData': {
                'segmentationValueType': segmentation_value_type,
                'definitions': definitions,
            }
        }
        if renames is not None:
            payload['renames'] = renames
        endpoint = f'{self.ENDPOINT_BASE}/{annotation_set_id}/segmentation-group'
        self._make_request('PUT', endpoint, json=payload)

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
