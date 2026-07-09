from ..entity_base_api import CreatableEntityApi, UpdatableEntityApi
from typing import TYPE_CHECKING, Any, Literal, overload
from datamint.entities.annotation_worklist import AnnotationWorklist
from typing_extensions import override
import logging
import warnings

if TYPE_CHECKING:
    from datamint.entities import Project
    from datamint.entities.resource import Resource

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
               project: 'str | Project | None' = None,
               return_url: str | None = None,
               *,
               project_id: str | None = None,
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
               project: 'str | Project | None' = None,
               return_url: str | None = None,
               *,
               project_id: str | None = None,
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
               project: 'str | Project | None' = None,
               return_url: str | None = None,
               *,
               project_id: str | None = None,
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
            project: Optional project ID or Project instance to associate with this worklist.
            return_url: Optional URL to redirect after annotation.
            project_id: (DEPRECATED) Use ``project`` instead.

        Returns:
            The ID of the created annotation set.
        """
        if project_id is not None:
            warnings.warn("The 'project_id' parameter is deprecated. "
                          "Please use 'project' instead", DeprecationWarning)
            if project is None:
                project = project_id

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
        if project is not None:
            payload['project_id'] = self._entid(project)
        if return_url is not None:
            payload['return_url'] = return_url
        return self._create(payload, return_entity=return_entity, exists_ok=exists_ok)

    def update_segmentation_group(self,
                                  worklist_id: 'str | AnnotationWorklist | None' = None,
                                  definitions: list[dict] | None = None,
                                  segmentation_value_type: str = 'single_label',
                                  renames: list[str] | None = None,
                                  *,
                                  annotation_worklist: 'str | AnnotationWorklist | None' = None) -> None:
        """Replace the segmentation-group definitions for an annotation worklist.

        Args:
            worklist_id: The annotation worklist ID or AnnotationWorklist instance.
            definitions: List of definition dicts with keys ``identifier``,
                ``color``, and ``index``.
            segmentation_value_type: ``'single_label'`` (default) or
                ``'multi_label'``.
            renames: Optional rename pairs (old→new identifier strings).
            annotation_worklist: (DEPRECATED) Use ``worklist_id`` instead.
        """
        if annotation_worklist is not None:
            warnings.warn("The 'annotation_worklist' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_worklist
        if worklist_id is None:
            raise TypeError("update_segmentation_group() missing required argument: 'worklist_id'")
        if definitions is None:
            raise TypeError("update_segmentation_group() missing required argument: 'definitions'")

        payload: dict = {
            'segmentationData': {
                'segmentationValueType': segmentation_value_type,
                'definitions': definitions,
            }
        }
        if renames is not None:
            payload['renames'] = renames
        self._make_entity_request('PUT', worklist_id,
                                  'segmentation-group',
                                  json=payload)

    def get_segmentation_group(self,
                               worklist_id: str | None = None,
                               *,
                               annotation_set: str | None = None) -> dict:
        """Get the segmentation group for a given worklist ID.

        Args:
            worklist_id: The annotation worklist ID.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("get_segmentation_group() missing required argument: 'worklist_id'")

        return self._make_entity_request('GET',
                                         worklist_id,
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
        worklist_id: str | None = None,
        identifier: str | None = None,
        *,
        annotation_set: str | None = None,
    ) -> None:
        """Delete a specific segmentation group from a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            identifier: The segmentation group identifier to delete.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("delete_segmentation_group() missing required argument: 'worklist_id'")
        if identifier is None:
            raise TypeError("delete_segmentation_group() missing required argument: 'identifier'")

        self._make_entity_request('DELETE', worklist_id,
                                  f'segmentation-group/{identifier}')

    def get_annotator_status(
        self,
        worklist_id: str | None = None,
        annotator_email: str | None = None,
        *,
        annotation_set: str | None = None,
        email: str | None = None,
    ) -> dict:
        """Get a specific annotator's progress status within a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            annotator_email: The annotator's email address.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            email: (DEPRECATED) Use ``annotator_email`` instead.

        Returns:
            Dict with annotator status information.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if email is not None:
            warnings.warn("The 'email' parameter is deprecated. "
                          "Please use 'annotator_email' instead", DeprecationWarning)
            if annotator_email is None:
                annotator_email = email
        if worklist_id is None:
            raise TypeError("get_annotator_status() missing required argument: 'worklist_id'")
        if annotator_email is None:
            raise TypeError("get_annotator_status() missing required argument: 'annotator_email'")

        return self._make_entity_request('GET', worklist_id,
                                         f'users/{annotator_email}').json()

    def get_segmentations(
        self,
        worklist_id: str | None = None,
        resource: 'str | Resource | None' = None,
        all: bool = True,
        annotator_email: str | None = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
        annotator: str | None = None,
    ) -> list[dict]:
        """Get all segmentations for a resource within a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            resource: The resource unique id or a Resource instance.
            all: Whether to get all segmentations.
            annotator_email: Optional annotator email filter.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.
            annotator: (DEPRECATED) Use ``annotator_email`` instead.

        Returns:
            List of segmentation objects.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if annotator is not None:
            warnings.warn("The 'annotator' parameter is deprecated. "
                          "Please use 'annotator_email' instead", DeprecationWarning)
            if annotator_email is None:
                annotator_email = annotator
        if worklist_id is None:
            raise TypeError("get_segmentations() missing required argument: 'worklist_id'")
        if resource is None:
            raise TypeError("get_segmentations() missing required argument: 'resource'")
        resource_id_str = self._entid(resource)

        params: dict[str, bool | str] = {'all': all}
        if annotator_email is not None:
            params['annotator'] = annotator_email
        return self._make_entity_request('GET', worklist_id,
                                         f'resources/{resource_id_str}/segmentations',
                                         params=params).json()

    def get_annotations(
        self,
        worklist_id: str | None = None,
        resource: 'str | Resource | None' = None,
        all: bool = True,
        annotator_email: str | None = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
        annotator: str | None = None,
    ) -> list[dict]:
        """Get all annotations (non-segmentation types) for a resource within a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            resource: The resource unique id or a Resource instance.
            all: Whether to get all annotations.
            annotator_email: Optional annotator email filter.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.
            annotator: (DEPRECATED) Use ``annotator_email`` instead.

        Returns:
            List of annotation objects.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if annotator is not None:
            warnings.warn("The 'annotator' parameter is deprecated. "
                          "Please use 'annotator_email' instead", DeprecationWarning)
            if annotator_email is None:
                annotator_email = annotator
        if worklist_id is None:
            raise TypeError("get_annotations() missing required argument: 'worklist_id'")
        if resource is None:
            raise TypeError("get_annotations() missing required argument: 'resource'")
        resource_id_str = self._entid(resource)

        params: dict[str, bool | str] = {'all': all}
        if annotator_email is not None:
            params['annotator'] = annotator_email
        return self._make_entity_request('GET', worklist_id,
                                         f'resources/{resource_id_str}/annotations',
                                         params=params).json()

    def get_ai_segmentations(
        self,
        worklist_id: str | None = None,
        resource: 'str | Resource | None' = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
    ) -> list[dict]:
        """Get AI-generated segmentations for a resource within a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            resource: The resource unique id or a Resource instance.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.

        Returns:
            List of AI segmentation objects.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if worklist_id is None:
            raise TypeError("get_ai_segmentations() missing required argument: 'worklist_id'")
        if resource is None:
            raise TypeError("get_ai_segmentations() missing required argument: 'resource'")
        resource_id_str = self._entid(resource)

        return self._make_entity_request('GET', worklist_id,
                                         f'resources/{resource_id_str}/ai_segmentations').json()

    def upload_annotations(
        self,
        worklist_id: str | None = None,
        resource: 'str | Resource | None' = None,
        payload: str | None = None,
        images: list[Any] | None = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
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
            worklist_id: The annotation worklist ID.
            resource: The resource unique id or a Resource instance.
            payload: A JSON string containing an array of segmentation items.
            images: Optional list of file-like objects or file paths to upload.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.

        Returns:
            Response dict with created annotation info.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if worklist_id is None:
            raise TypeError("upload_annotations() missing required argument: 'worklist_id'")
        if resource is None:
            raise TypeError("upload_annotations() missing required argument: 'resource'")
        if payload is None:
            raise TypeError("upload_annotations() missing required argument: 'payload'")
        resource_id_str = self._entid(resource)

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
                return self._make_entity_request('POST', worklist_id,
                                                 f'resources/{resource_id_str}/segmentations',
                                                 files=files, data={'payload': payload}).json()
            finally:
                for f in opened_files:
                    f.close()
        else:
            return self._make_entity_request('POST', worklist_id,
                                             f'resources/{resource_id_str}/segmentations',
                                             data={'payload': payload}).json()

    def update_annotation_status(
        self,
        worklist_id: str | None = None,
        resource: 'str | Resource | None' = None,
        status: Literal['opened', 'annotated', 'closed', 'approved', 'revision_request'] = 'closed',
        annotator_email: str | None = None,
        message: str | None = None,
        path: str | None = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
        annotator: str | None = None,
    ) -> list[dict]:
        """Update the annotation status for a resource within a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            resource: The resource unique id or a Resource instance.
            status: New status (opened, annotated, closed, approved, revision_request).
            annotator_email: Optional annotator email.
            message: Optional message.
            path: Optional path.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.
            annotator: (DEPRECATED) Use ``annotator_email`` instead.

        Returns:
            List of objects with updated status information.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if annotator is not None:
            warnings.warn("The 'annotator' parameter is deprecated. "
                          "Please use 'annotator_email' instead", DeprecationWarning)
            if annotator_email is None:
                annotator_email = annotator
        if worklist_id is None:
            raise TypeError("update_annotation_status() missing required argument: 'worklist_id'")
        if resource is None:
            raise TypeError("update_annotation_status() missing required argument: 'resource'")
        resource_id_str = self._entid(resource)

        payload = {'status': status}
        if annotator_email is not None:
            payload['annotator'] = annotator_email
        if message is not None:
            payload['message'] = message
        if path is not None:
            payload['path'] = path
        return self._make_entity_request('POST', worklist_id,
                                         f'resources/{resource_id_str}/status',
                                         json=payload).json()

    def set_annotator(
        self,
        worklist_id: str | None = None,
        user_id: str | None = None,
        status: Literal['active', 'frozen'] | None = None,
        expertise_level: Literal['learner', 'trained', 'expert'] | None = None,
        return_url: str | None = None,
        *,
        annotation_set: str | None = None,
    ) -> dict:
        """Set or update an annotator's status and expertise level in a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            user_id: The user's UUID.
            status: Annotator status (active or frozen).
            expertise_level: Expertise level (learner, trained, or expert).
            return_url: Optional return URL.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.

        Returns:
            Response dict with created annotator info.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("set_annotator() missing required argument: 'worklist_id'")
        if user_id is None:
            raise TypeError("set_annotator() missing required argument: 'user_id'")
        if status is None:
            raise TypeError("set_annotator() missing required argument: 'status'")
        if expertise_level is None:
            raise TypeError("set_annotator() missing required argument: 'expertise_level'")

        payload = {
            'status': status,
            'expertise_level': expertise_level,
        }
        if return_url is not None:
            payload['return_url'] = return_url
        return self._make_entity_request('POST', worklist_id,
                                         f'annotators/{user_id}',
                                         json=payload).json()

    def remove_annotator(
        self,
        worklist_id: str | None = None,
        user_id: str | None = None,
        *,
        annotation_set: str | None = None,
    ) -> None:
        """Remove an annotator from a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            user_id: The user's UUID.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("remove_annotator() missing required argument: 'worklist_id'")
        if user_id is None:
            raise TypeError("remove_annotator() missing required argument: 'user_id'")

        self._make_request('DELETE',
                           f'/annotationsets/{worklist_id}/annotators/{user_id}')

    def update_resources(
        self,
        worklist_id: str | None = None,
        resource_ids_to_add: list[str] | None = None,
        resource_ids_to_delete: list[str] | None = None,
        *,
        annotation_set: str | None = None,
        resources_to_add: list[str] | None = None,
        resources_to_delete: list[str] | None = None,
    ) -> dict:
        """Add or remove resources from a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            resource_ids_to_add: Optional list of resource IDs to add.
            resource_ids_to_delete: Optional list of resource IDs to delete.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resources_to_add: (DEPRECATED) Use ``resource_ids_to_add`` instead.
            resources_to_delete: (DEPRECATED) Use ``resource_ids_to_delete`` instead.

        Returns:
            Response dict with updated worklist info.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resources_to_add is not None:
            warnings.warn("The 'resources_to_add' parameter is deprecated. "
                          "Please use 'resource_ids_to_add' instead", DeprecationWarning)
            if resource_ids_to_add is None:
                resource_ids_to_add = resources_to_add
        if resources_to_delete is not None:
            warnings.warn("The 'resources_to_delete' parameter is deprecated. "
                          "Please use 'resource_ids_to_delete' instead", DeprecationWarning)
            if resource_ids_to_delete is None:
                resource_ids_to_delete = resources_to_delete
        if worklist_id is None:
            raise TypeError("update_resources() missing required argument: 'worklist_id'")

        payload: dict = {}
        if resource_ids_to_add is not None:
            payload['resource_ids_to_add'] = resource_ids_to_add
        if resource_ids_to_delete is not None:
            payload['resource_ids_to_delete'] = resource_ids_to_delete
        return self._make_entity_request('POST', worklist_id, 'resources',
                                         json=payload).json()

    def get_annotation_statuses(
        self,
        worklist_id: str | None = None,
        status: str | None = None,
        user_id: str | None = None,
        resource: 'str | Resource | None' = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
    ) -> list[dict]:
        """Get annotation statuses for a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            status: Optional status filter. Allowed values: ``opened``, ``annotated``,
                ``closed``, ``approved``, ``revision_request``.
            user_id: Optional user ID filter.
            resource: Optional resource unique id, or Resource instance, filter.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.

        Returns:
            List of annotation status dicts.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if worklist_id is None:
            raise TypeError("get_annotation_statuses() missing required argument: 'worklist_id'")

        params: dict[str, str | list[str]] = {}
        if status is not None:
            params['status'] = status
        if user_id is not None:
            params['user_id'] = user_id
        if resource is not None:
            params['resource_id'] = self._entid(resource)
        return self._make_entity_request('GET', worklist_id,
                                         'annotation-statuses',
                                         params=params or None).json()

    def reset_annotator_status(
        self,
        worklist_id: str | None = None,
        resource: 'str | Resource | None' = None,
        annotator_email: str | None = None,
        *,
        annotation_set: str | None = None,
        resource_id: str | None = None,
        annotator: str | None = None,
    ) -> None:
        """Reset one annotator's resource status and delete related annotations.

        Args:
            worklist_id: The annotation worklist ID.
            resource: The resource unique id or a Resource instance.
            annotator_email: The annotator's email address.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            resource_id: (DEPRECATED) Use ``resource`` instead.
            annotator: (DEPRECATED) Use ``annotator_email`` instead.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if resource_id is not None:
            warnings.warn("The 'resource_id' parameter is deprecated. "
                          "Please use 'resource' instead", DeprecationWarning)
            if resource is None:
                resource = resource_id
        if annotator is not None:
            warnings.warn("The 'annotator' parameter is deprecated. "
                          "Please use 'annotator_email' instead", DeprecationWarning)
            if annotator_email is None:
                annotator_email = annotator
        if worklist_id is None:
            raise TypeError("reset_annotator_status() missing required argument: 'worklist_id'")
        if resource is None:
            raise TypeError("reset_annotator_status() missing required argument: 'resource'")
        if annotator_email is None:
            raise TypeError("reset_annotator_status() missing required argument: 'annotator_email'")
        resource_id_str = self._entid(resource)

        self._make_entity_request('DELETE', worklist_id,
                                   f'resources/{resource_id_str}/annotator/{annotator_email}/status-reset')

    def get_annotators_statistics(
        self,
        worklist_id: str | None = None,
        annotator_email: str | None = None,
        *,
        annotation_set: str | None = None,
        email: str | None = None,
    ) -> list[dict]:
        """Get annotator statistics for a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            annotator_email: Optional annotator email filter.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.
            email: (DEPRECATED) Use ``annotator_email`` instead.

        Returns:
            List of per-annotator stat dicts.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if email is not None:
            warnings.warn("The 'email' parameter is deprecated. "
                          "Please use 'annotator_email' instead", DeprecationWarning)
            if annotator_email is None:
                annotator_email = email
        if worklist_id is None:
            raise TypeError("get_annotators_statistics() missing required argument: 'worklist_id'")

        params: dict[str, str] | None = None
        if annotator_email is not None:
            params = {'email': annotator_email}
        return self._make_entity_request('GET', worklist_id,
                                         'annotators-statistic',
                                         params=params).json()

    def get_annotations_statistics(
        self,
        worklist_id: str | None = None,
        *,
        annotation_set: str | None = None,
    ) -> list[dict]:
        """Get annotation statistics for a worklist.

        Args:
            worklist_id: The annotation worklist ID.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.

        Returns:
            List of annotation stat objects.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("get_annotations_statistics() missing required argument: 'worklist_id'")

        return self._make_entity_request('GET', worklist_id,
                                         'annotations-statistic').json()

    def download_annotations(
        self,
        worklist_id: str | None = None,
        from_date: str | None = None,
        to_date: str | None = None,
        annotators: list[str] | None = None,
        annotations: list[str] | None = None,
        format: str = 'csv',
        *,
        annotation_set: str | None = None,
    ) -> bytes:
        """Download annotations as a streamed file.

        Args:
            worklist_id: The annotation worklist ID.
            from_date: Optional start date filter (ISO string).
            to_date: Optional end date filter (ISO string).
            annotators: Optional list of annotator emails. Accepts either a repeated/array
                input or a comma-separated string.
            annotations: Optional list of annotation identifiers. Accepts either a repeated/array
                input or a comma-separated string.
            format: Export format. Allowed values: ``csv`` (default), ``excel``.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.

        Returns:
            The raw bytes of the downloaded file.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("download_annotations() missing required argument: 'worklist_id'")

        params: dict[str, str | list[str]] = {'format': format}
        if from_date is not None:
            params['from'] = from_date
        if to_date is not None:
            params['to'] = to_date
        if annotators is not None:
            params['annotators[]'] = annotators
        if annotations is not None:
            params['annotations[]'] = annotations
        response = self._make_entity_request('GET', worklist_id,
                                             'download-annotations',
                                             params=params or None)
        return response.content

    def upload_segmentation_group(
        self,
        worklist_id: str | None = None,
        file: Any = None,
        replace_existing: bool = True,
        *,
        annotation_set: str | None = None,
    ) -> dict:
        """Upload a segmentation group definition file to an annotation set.

        Args:
            worklist_id: The annotation worklist ID.
            file: The file to upload. Must have one of these extensions:
                ``.yaml``, ``.yml``, ``.csv``, ``.json``. Maximum size: 10 MB.
            replace_existing: Whether to replace existing segmentation data.
            annotation_set: (DEPRECATED) Use ``worklist_id`` instead.

        Returns:
            Response dict with created segmentation group info.
        """
        if annotation_set is not None:
            warnings.warn("The 'annotation_set' parameter is deprecated. "
                          "Please use 'worklist_id' instead", DeprecationWarning)
            if worklist_id is None:
                worklist_id = annotation_set
        if worklist_id is None:
            raise TypeError("upload_segmentation_group() missing required argument: 'worklist_id'")
        if file is None:
            raise TypeError("upload_segmentation_group() missing required argument: 'file'")

        # Use multipart form data for file upload
        if hasattr(file, 'read'):
            file_data = file.read()
            filename = getattr(file, 'name', 'segmentation.yaml')
        else:
            file_data = file
            filename = getattr(file, 'name', 'segmentation.yaml')

        files = {'file': (filename, file_data)}
        data = {'replace_existing': str(replace_existing).lower()}
        return self._make_entity_request('POST', worklist_id,
                                         'segmentation-group/upload',
                                         files=files, data=data).json()
