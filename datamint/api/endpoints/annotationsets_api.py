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
               annotators: list[dict] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               segmentation_data: dict | None = None,
               viewable_ai_annotations: list[str] | None = None,
               editable_ai_annotations: list[str] | None = None,
               project_id: str | None = None,
               return_url: str | None = None,
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
               annotators: list[dict] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               segmentation_data: dict | None = None,
               viewable_ai_annotations: list[str] | None = None,
               editable_ai_annotations: list[str] | None = None,
               project_id: str | None = None,
               return_url: str | None = None,
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
               annotators: list[dict] | None = None,
               frame_labels: list[str] | None = None,
               image_labels: list[str] | None = None,
               segmentation_data: dict | None = None,
               viewable_ai_annotations: list[str] | None = None,
               editable_ai_annotations: list[str] | None = None,
               project_id: str | None = None,
               return_url: str | None = None,
               *,
               return_entity: bool = True,
               exists_ok: bool = False,
               ) -> str | AnnotationWorklist:
        """Create a new annotation worklist.

        Args:
            name: Name of the annotation worklist.
            resource_ids: List of resource IDs to include.
            description: Optional description.
            annotations: Optional list of annotation spec dicts. Each dict should have:
                - ``type`` (required): annotation type enum.
                - ``identifier`` (optional): non-empty string.
                - ``required`` (required): boolean.
                - ``scope`` (required): scope enum.
                - ``values`` (optional): string array.
            annotators: Optional list of annotator dicts. Each dict should have:
                - ``email`` (required): valid email.
                - ``expertise_level`` (optional): one of ``learner``, ``trained``, ``expert``.
            frame_labels: Optional list of frame label names.
            image_labels: Optional list of image label names.
            segmentation_data: Optional segmentation group definition dict with keys:
                - ``segmentationValueType`` (required): one of ``single_label``, ``multi_label``, ``float``.
                - ``definitions`` (required): array of definition dicts.
            viewable_ai_annotations: Optional list of AI annotation identifiers to display.
            editable_ai_annotations: Optional list of AI annotation identifiers to allow editing.
            project_id: Optional project UUID to associate with this worklist.
            return_url: Optional URL to redirect after annotation.

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
        if segmentation_data is not None:
            payload['segmentation_data'] = segmentation_data
        if viewable_ai_annotations is not None:
            payload['viewable_ai_annotations'] = viewable_ai_annotations
        if editable_ai_annotations is not None:
            payload['editable_ai_annotations'] = editable_ai_annotations
        if project_id is not None:
            payload['project_id'] = project_id
        if return_url is not None:
            payload['return_url'] = return_url
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

    def delete_segmentation_group(
        self,
        annotation_set: str,
        identifier: str,
    ) -> None:
        """Delete a specific segmentation group from a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            identifier: The segmentation group identifier to delete.
        """
        self._make_entity_request('DELETE', annotation_set,
                                  f'segmentation-group/{identifier}')

    def get_annotator_status(
        self,
        annotation_set: str,
        email: str,
    ) -> dict:
        """Get a specific annotator's progress status within a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            email: The annotator's email address.
            
        Returns:
            Dict with annotator status information.
        """
        return self._make_entity_request('GET', annotation_set,
                                         f'users/{email}').json()

    def get_segmentations(
        self,
        annotation_set: str,
        resource_id: str,
        all: bool = True,
        annotator: str | None = None,
    ) -> list[dict]:
        """Get all segmentations for a resource within a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            resource_id: The resource ID.
            all: Whether to get all segmentations.
            annotator: Optional annotator email filter.
            
        Returns:
            List of segmentation objects.
        """
        params: dict[str, bool | str] = {'all': all}
        if annotator is not None:
            params['annotator'] = annotator
        return self._make_entity_request('GET', annotation_set,
                                         f'resources/{resource_id}/segmentations',
                                         params=params).json()

    def get_annotations(
        self,
        annotation_set: str,
        resource_id: str,
        all: bool = True,
        annotator: str | None = None,
    ) -> list[dict]:
        """Get all annotations (non-segmentation types) for a resource within a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            resource_id: The resource ID.
            all: Whether to get all annotations.
            annotator: Optional annotator email filter.
            
        Returns:
            List of annotation objects.
        """
        params: dict[str, bool | str] = {'all': all}
        if annotator is not None:
            params['annotator'] = annotator
        return self._make_entity_request('GET', annotation_set,
                                         f'resources/{resource_id}/annotations',
                                         params=params).json()

    def get_ai_segmentations(
        self,
        annotation_set: str,
        resource_id: str,
    ) -> list[dict]:
        """Get AI-generated segmentations for a resource within a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            resource_id: The resource ID.
            
        Returns:
            List of AI segmentation objects.
        """
        return self._make_entity_request('GET', annotation_set,
                                         f'resources/{resource_id}/ai_segmentations').json()

    def upload_annotations(
        self,
        annotation_set: str,
        resource_id: str,
        payload: str,
        images: list[Any] | None = None,
    ) -> dict:
        """Upload one or more resource segmentations.

        The endpoint expects multipart form data with:
        - ``payload``: required JSON string containing a JSON array of segmentation items.
        - ``images``: optional file field (up to 100 files).

        Each payload item shape:
        - ``id``: UUID, optional.
        - ``identifier``: string, expected.
        - ``frame_index``: number, expected.
        - ``width``: number, expected.
        - ``height``: number, expected.
        - ``set_name``: string or null, optional.
        - ``geometry``: any value, optional.

        Args:
            annotation_set: The annotation set ID.
            resource_id: The resource ID.
            payload: A JSON string containing an array of segmentation items.
            images: Optional list of file-like objects or file paths to upload.

        Returns:
            Response dict with created annotation info.
        """
        if images is not None:
            files = {}
            opened_files: list = []
            try:
                for i, img in enumerate(images):
                    if hasattr(img, 'read'):
                        filename = getattr(img, 'name', f'image_{i}')
                        files[f'images'] = (filename, img)
                    else:
                        f = open(img, 'rb')
                        opened_files.append(f)
                        files[f'images'] = (img, f)
                return self._make_entity_request('POST', annotation_set,
                                                 f'resources/{resource_id}/segmentations',
                                                 files=files, data={'payload': payload}).json()
            finally:
                for f in opened_files:
                    f.close()
        else:
            return self._make_entity_request('POST', annotation_set,
                                             f'resources/{resource_id}/segmentations',
                                             data={'payload': payload}).json()

    def update_annotation_status(
        self,
        annotation_set: str,
        resource_id: str,
        status: Literal['opened', 'annotated', 'closed', 'approved', 'revision_request'] = 'closed',
        annotator: str | None = None,
        message: str | None = None,
        path: str | None = None,
    ) -> list[dict]:
        """Update the annotation status for a resource within a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            resource_id: The resource ID.
            status: New status (opened, annotated, closed, approved, revision_request).
            annotator: Optional annotator email.
            message: Optional message.
            path: Optional path.
            
        Returns:
            List of objects with updated status information.
        """
        payload = {'status': status}
        if annotator is not None:
            payload['annotator'] = annotator
        if message is not None:
            payload['message'] = message
        if path is not None:
            payload['path'] = path
        return self._make_entity_request('POST', annotation_set,
                                         f'resources/{resource_id}/status',
                                         json=payload).json()

    def set_annotator(
        self,
        annotation_set: str,
        user_id: str,
        status: Literal['active', 'frozen'],
        expertise_level: Literal['learner', 'trained', 'expert'],
        return_url: str | None = None,
    ) -> dict:
        """Set or update an annotator's status and expertise level in a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            user_id: The user's UUID.
            status: Annotator status (active or frozen).
            expertise_level: Expertise level (learner, trained, or expert).
            return_url: Optional return URL.
            
        Returns:
            Response dict with created annotator info.
        """
        payload = {
            'status': status,
            'expertise_level': expertise_level,
        }
        if return_url is not None:
            payload['return_url'] = return_url
        return self._make_entity_request('POST', annotation_set,
                                         f'annotators/{user_id}',
                                         json=payload).json()

    def remove_annotator(
        self,
        annotation_set: str,
        user_id: str,
    ) -> None:
        """Remove an annotator from a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            user_id: The user's UUID.
        """
        self._make_request('DELETE',
                           f'/annotationsets/{annotation_set}/annotators/{user_id}')

    def update_resources(
        self,
        annotation_set: str,
        resources_to_add: list[str] | None = None,
        resources_to_delete: list[str] | None = None,
    ) -> dict:
        """Add or remove resources from a worklist.
        
        Args:
            annotation_set: The annotation set ID.
            resources_to_add: Optional list of resource IDs to add.
            resources_to_delete: Optional list of resource IDs to delete.
            
        Returns:
            Response dict with updated worklist info.
        """
        payload: dict = {}
        if resources_to_add is not None:
            payload['resource_ids_to_add'] = resources_to_add
        if resources_to_delete is not None:
            payload['resource_ids_to_delete'] = resources_to_delete
        return self._make_entity_request('POST', annotation_set, 'resources',
                                         json=payload).json()

    def get_annotation_statuses(
        self,
        annotation_set: str,
        status: str | None = None,
        user_id: str | None = None,
        resource_id: str | None = None,
    ) -> list[dict]:
        """Get annotation statuses for a worklist.

        Args:
            annotation_set: The annotation set ID.
            status: Optional status filter. Allowed values: ``opened``, ``annotated``,
                ``closed``, ``approved``, ``revision_request``.
            user_id: Optional user ID filter.
            resource_id: Optional resource ID filter (UUID v4).

        Returns:
            List of annotation status dicts.
        """
        params: dict[str, str | list[str]] = {}
        if status is not None:
            params['status'] = status
        if user_id is not None:
            params['user_id'] = user_id
        if resource_id is not None:
            params['resource_id'] = resource_id
        return self._make_entity_request('GET', annotation_set,
                                         'annotation-statuses',
                                         params=params or None).json()

    def reset_annotator_status(
        self,
        annotation_set: str,
        resource_id: str,
        annotator: str,
    ) -> None:
        """Reset one annotator's resource status and delete related annotations.

        Args:
            annotation_set: The annotation set ID.
            resource_id: The resource ID.
            annotator: The annotator identifier.
        """
        self._make_entity_request('DELETE', annotation_set,
                                   f'resources/{resource_id}/annotator/{annotator}/status-reset')

    def get_annotators_statistics(
        self,
        annotation_set: str,
        email: str | None = None,
    ) -> list[dict]:
        """Get annotator statistics for a worklist.

        Args:
            annotation_set: The annotation set ID.
            email: Optional annotator email filter.

        Returns:
            List of per-annotator stat dicts.
        """
        params: dict[str, str] | None = None
        if email is not None:
            params = {'email': email}
        return self._make_entity_request('GET', annotation_set,
                                         'annotators-statistic',
                                         params=params).json()

    def get_annotations_statistics(
        self,
        annotation_set: str,
    ) -> list[dict]:
        """Get annotation statistics for a worklist.

        Args:
            annotation_set: The annotation set ID.

        Returns:
            List of annotation stat objects.
        """
        return self._make_entity_request('GET', annotation_set,
                                         'annotations-statistic').json()

    def download_annotations(
        self,
        annotation_set: str,
        from_date: str | None = None,
        to_date: str | None = None,
        annotators: list[str] | None = None,
        annotations: list[str] | None = None,
        format: str = 'csv',
    ) -> bytes:
        """Download annotations as a streamed file.

        Args:
            annotation_set: The annotation set ID.
            from_date: Optional start date filter (ISO string).
            to_date: Optional end date filter (ISO string).
            annotators: Optional list of annotator emails. Accepts either a repeated/array
                input or a comma-separated string.
            annotations: Optional list of annotation identifiers. Accepts either a repeated/array
                input or a comma-separated string.
            format: Export format. Allowed values: ``csv`` (default), ``excel``.

        Returns:
            The raw bytes of the downloaded file.
        """
        params: dict[str, str | list[str]] = {'format': format}
        if from_date is not None:
            params['from'] = from_date
        if to_date is not None:
            params['to'] = to_date
        if annotators is not None:
            params['annotators[]'] = annotators
        if annotations is not None:
            params['annotations[]'] = annotations
        response = self._make_entity_request('GET', annotation_set,
                                             'download-annotations',
                                             params=params or None)
        return response.content

    def upload_segmentation_group(
        self,
        annotation_set: str,
        file: Any,
        replace_existing: bool = True,
    ) -> dict:
        """Upload a segmentation group definition file to an annotation set.

        Args:
            annotation_set: The annotation set ID.
            file: The file to upload. Must have one of these extensions:
                ``.yaml``, ``.yml``, ``.csv``, ``.json``. Maximum size: 10 MB.
            replace_existing: Whether to replace existing segmentation data.

        Returns:
            Response dict with created segmentation group info.
        """
        # Use multipart form data for file upload
        import io
        if hasattr(file, 'read'):
            file_data = file.read()
            filename = getattr(file, 'name', 'segmentation.yaml')
        else:
            file_data = file
            filename = getattr(file, 'name', 'segmentation.yaml')
        
        files = {'file': (filename, file_data)}
        data = {'replace_existing': str(replace_existing).lower()}
        return self._make_entity_request('POST', annotation_set,
                                         'segmentation-group/upload',
                                         files=files, data=data).json()
