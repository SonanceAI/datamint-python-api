from ..entity_base_api import CreatableEntityApi, UpdatableEntityApi
from typing import TYPE_CHECKING, Any, Literal, overload
from datamint.entities.annotation_worklist import AnnotationWorklist
from typing_extensions import override
import logging

if TYPE_CHECKING:
    from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)


class AnnotationWorklistApi(CreatableEntityApi[AnnotationWorklist],
                            UpdatableEntityApi[AnnotationWorklist]):

    def __init__(self, config: Any, client: Any = None) -> None:
        super().__init__(config, AnnotationWorklist, 'annotationsets', client)

    @overload
    def create(self,
               name: str,
               resource_ids: list[str],
               description: str | None = None,
               annotations: list[dict] | None = None,
               annotators: list[str] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               *,
               return_entity: Literal[True] = True,
               exists_ok: bool = False
               ) -> AnnotationWorklist: ...

    @overload
    def create(self,
               name: str,
               resource_ids: list[str],
               description: str | None = None,
               annotations: list[dict] | None = None,
               annotators: list[str] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               *,
               return_entity: Literal[False],
               exists_ok: bool = False
               ) -> str: ...

    @override
    def create(self,
               name: str,
               resource_ids: list[str],
               description: str | None = None,
               annotations: list[dict] | None = None,
               annotators: list[str] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               *,
               return_entity: bool = True,
               exists_ok: bool = False,
               ) -> str | AnnotationWorklist:
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
        return self._create(payload, return_entity=return_entity, exists_ok=exists_ok)

    def update_segmentation_group(self,
                                  annotation_worklist: str | AnnotationWorklist,
                                  definitions: list[dict],
                                  segmentation_value_type: str = 'single_label',
                                  renames: list[str] | None = None) -> None:
        """Replace the segmentation-group definitions for an annotation worklist.

        Args:
            annotation_worklist: The annotation worklist ID or AnnotationWorklist instance.
            definitions: List of definition dicts with keys ``identifier``,
                ``color``, and ``index``.
            segmentation_value_type: ``'single_label'`` (default) or
                ``'multi_label'``.
            renames: Optional rename pairs (old→new identifier strings).
        """
        payload: dict = {
            'segmentationData': {
                'segmentationValueType': segmentation_value_type,
                'definitions': definitions,
            }
        }
        if renames is not None:
            payload['renames'] = renames
        self._make_entity_request('PUT', annotation_worklist,
                                  'segmentation-group',
                                  json=payload)

    def get_segmentation_group(self, annotation_set: str) -> dict:
        """Get the segmentation group for a given annotation set ID."""

        return self._make_entity_request('GET',
                                         annotation_set,
                                         'segmentation-group').json()

    def get_by_project(self, project: 'str | Project') -> list[AnnotationWorklist]:
        """Get the worklist IDs associated with a project.

        Args:
            project: The project ID or Project instance.

        Returns:
            List of worklist IDs.
        """
        proj_id = CreatableEntityApi._entid(project)
        response = self._make_request('GET',
                                      endpoint=f'projects/{proj_id}/worklists')
        ret = []
        remap_keys = {'worklist_id': 'id', 'worklist_name': 'name'}
        for item in response.json():
            for old_key, new_key in remap_keys.items():
                if old_key in item:
                    if new_key in item:
                        _LOGGER.warning(f"Key conflict when remapping '{old_key}' to '{new_key}' in response item: "
                                        " both keys are present. Keeping original keys.")
                    else:
                        item[new_key] = item.pop(old_key)
            ret.append(self._init_entity_obj(**item))
        return ret
