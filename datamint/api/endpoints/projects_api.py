from typing import Literal, TYPE_CHECKING, overload
from collections.abc import Sequence
from pathlib import Path

from ..entity_base_api import ApiConfig, CRUDEntityApi
from datamint.entities.project import Project
from datamint.entities.project_resource_split import ProjectResourceSplit
import httpx
from datamint.entities.resource import Resource
from datamint.exceptions import EntityAlreadyExistsError, ItemNotFoundError
if TYPE_CHECKING:
    from . import AnnotationSetsApi, ResourcesApi
    from datamint.entities.annotations.annotation_spec import AnnotationSpec


class ProjectsApi(CRUDEntityApi[Project]):
    """API handler for project-related endpoints."""

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None,
                 resources_api: 'ResourcesApi | None' = None) -> None:
        """Initialize the projects API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        from . import AnnotationSetsApi, ResourcesApi

        super().__init__(config, Project, 'projects', client)
        self.resources_api = resources_api or ResourcesApi(config, client, projects_api=self)
        self.annotationsets_api = AnnotationSetsApi(config, client)

    def get_project_resources(self, project: Project | str) -> list[Resource]:
        """Get resources associated with a specific project.

        Args:
            project: The ID or instance of the project to fetch resources for.

        Returns:
            A list of resource instances associated with the project.
        """
        response = self._get_child_entities(project, 'resources')
        resources_data = response.json()
        resources = [self.resources_api._init_entity_obj(**item) for item in resources_data]
        return resources

    @overload
    def create(self,
               name: str,
               description: str,
               resources_ids: list[str] | None = None,
               is_active_learning: bool = False,
               two_up_display: bool = False,
               segmentation_spec: Literal['single_label', 'multi_label'] = 'single_label',
               *,
               return_entity: Literal[True] = True,
               exists_ok: bool = False
               ) -> Project: ...

    @overload
    def create(self,
               name: str,
               description: str,
               resources_ids: list[str] | None = None,
               is_active_learning: bool = False,
               two_up_display: bool = False,
               segmentation_spec: Literal['single_label', 'multi_label'] = 'single_label',
               *,
               return_entity: Literal[False],
               exists_ok: bool = False
               ) -> str: ...

    def create(self,
               name: str,
               description: str,
               resources_ids: list[str] | None = None,
               is_active_learning: bool = False,
               two_up_display: bool = False,
               segmentation_spec: Literal['single_label', 'multi_label'] = 'single_label',
               *,
               return_entity: bool = True,
               exists_ok: bool = False
               ) -> str | Project:
        """Create a new project.

        Args:
            name: The name of the project.
            description: The description of the project.
            resources_ids: The list of resource ids to be included in the project.
            is_active_learning: Whether the project is an active learning project or not.
            two_up_display: Allow annotators to display multiple resources for annotation.
            return_entity: Whether to return the created Project instance or just its ID.
            exists_ok: If ``True``, do not raise an error when a project with the same
                name already exists. Instead, the existing project is returned when
                possible.

        Returns:
            The id of the created project.
        """
        proj = self.get_by_name(name, include_archived=True)
        if proj is not None:
            if exists_ok:
                return proj if return_entity else proj.id
            else:
                raise EntityAlreadyExistsError(entity_type='Project', params={'name': name})
        resources_ids = resources_ids or []

        project_data = {'name': name,
                        'is_active_learning': is_active_learning,
                        'resource_ids': resources_ids,
                        "segmentationData": { "segmentationValueType": segmentation_spec, "definitions": [] },
                        'annotation_set': {
                            "resource_ids": resources_ids,
                            "annotations": [],
                        },
                        "two_up_display": two_up_display,
                        "require_review": False,
                        'description': description}

        return self._create(project_data, return_entity=return_entity, exists_ok=exists_ok)  # type: ignore[return-value]

    def get_all(self, limit: int | None = None) -> Sequence[Project]:
        """Get all projects.

        Args:
            limit: The maximum number of projects to return. If None, return all projects.

        Returns:
            A list of project instances.
        """
        return self.get_list(limit=limit, params={'includeArchived': True})

    def get_by_name(self,
                    name: str,
                    include_archived: bool = True) -> Project | None:
        """Get a project by its name.

        Args:
            name (str): The name of the project.
            include_archived (bool): Whether to include archived projects in the search.

        Returns:
            The project instance if found, otherwise None.
        """
        if include_archived:
            projects = self.get_list(params={'includeArchived': True})
        else:
            projects = self.get_all()
        for project in projects:
            if project.name == name:
                return project
        return None

    def _get_by_name_or_id(self, project: str) -> Project | None:
        """Get a project by its name or ID.

        Args:
            project (str): The name or ID of the project.

        Returns:
            The project instance if found, otherwise None.
        """
        projects = self.get_all()
        for proj in projects:
            if proj.name == project or proj.id == project:
                return proj
        return None

    def add_resources(self,
                      resources: str | Sequence[str] | Resource | Sequence[Resource],
                      project: str | Project,
                      ) -> None:
        """
        Add resources to a project.

        Args:
            resources: The resource unique id or a list of resource unique ids.
            project: The project name, id or :class:`Project` object to add the resource to.
        """
        if isinstance(resources, str):
            resources_ids = [resources]
        elif isinstance(resources, Resource):
            resources_ids = [resources.id]
        else:
            resources_ids = [res if isinstance(res, str) else res.id for res in resources]

        if isinstance(project, str):
            if len(project) == 36:
                project_id = project
            else:
                # get the project id by its name
                project_found = self._get_by_name_or_id(project)
                if project_found is None:
                    raise ValueError(f"Project '{project}' not found.")
                project_id = project_found.id
        else:
            project_id = project.id

        self._make_entity_request('POST', project_id, add_path='resources',
                                  json={'resource_ids_to_add': resources_ids, 'all_files_selected': False})

    # def download(self, project: str | Project,
    #              outpath: str,
    #              all_annotations: bool = False,
    #              include_unannotated: bool = False,
    #              ) -> None:
    #     """Download a project by its id.

    #     Args:
    #         project: The project id or Project instance.
    #         outpath: The path to save the project zip file.
    #         all_annotations: Whether to include all annotations in the downloaded dataset,
    #             even those not made by the provided project.
    #         include_unannotated: Whether to include unannotated resources in the downloaded dataset.
    #     """
    #     from tqdm.auto import tqdm
    #     params = {'all_annotations': all_annotations}
    #     if include_unannotated:
    #         params['include_unannotated'] = include_unannotated

    #     project_id = self._entid(project)
    #     with self._stream_entity_request('GET', project_id,
    #                                      add_path='annotated_dataset',
    #                                      params=params) as response:
    #         total_size = int(response.headers.get('content-length', 0))
    #         if total_size == 0:
    #             total_size = None
    #         with tqdm(total=total_size, unit='B', unit_scale=True) as progress_bar:
    #             with open(outpath, 'wb') as file:
    #                 for data in response.iter_bytes(1024):
    #                     progress_bar.update(len(data))
    #                     file.write(data)

    def set_work_status(self,
                        project: str | Project,
                        resource: str | Resource,
                        status: Literal['opened', 'annotated', 'closed']) -> None:
        """
        Set the status of a resource.

        Args:
            project: The project unique id or a project object.
            resource: The resource unique id or a resource object.
            status: The new status to set.
        """
        resource_id = self._entid(resource)
        proj_id = self._entid(project)

        jsondata = {
            'status': status
        }
        self._make_entity_request('POST',
                                  entity_id=proj_id,
                                  add_path=f'resources/{resource_id}/status',
                                  json=jsondata)

    def get_annotations_specs(self, project: str | Project) -> Sequence['AnnotationSpec']:
        """Get the annotations specs for a given project.

        Args:
            project: The project id or Project instance.

        Returns:
            A sequence of AnnotationSpec instances.
        """

        if isinstance(project, str):
            project = self.get_by_id(project)
        return self.annotationsets_api.get_annotations_specs(project)

    # ------------------------------------------------------------------
    # Project members
    # ------------------------------------------------------------------

    def get_members(self, project: str | Project) -> list[dict]:
        """List all members of a project with their roles.

        Args:
            project: The project ID or Project instance.

        Returns:
            List of member dicts (keys: ``project_id``, ``user_id``, ``roles``,
            ``expertise_level``, ``status``, ``firstname``, ``lastname``).
        """
        response = self._make_entity_request('GET', project, add_path='members')
        return response.json()

    def set_member(self,
                   project: str | Project,
                   user_id: str,
                   roles: list[str]) -> None:
        """Set (or update) a user's roles in a project.

        Args:
            project: The project ID or Project instance.
            user_id: The user's UUID.
            roles: List of role strings, e.g. ``['PROJECT_ANNOTATOR']``.
        """
        proj_id = self._entid(project)
        self._make_request('POST', f'/{self.endpoint_base}/{proj_id}/members/{user_id}',
                           json={'roles': roles})

    def remove_member(self,
                      project: str | Project,
                      user_id: str) -> None:
        """Remove a user from a project.

        Args:
            project: The project ID or Project instance.
            user_id: The user's UUID.
        """
        proj_id = self._entid(project)
        self._make_request('DELETE', f'/{self.endpoint_base}/{proj_id}/members/{user_id}')

    # ------------------------------------------------------------------
    # Worklist
    # ------------------------------------------------------------------

    def get_worklist(self, project: str | Project) -> dict:
        """Get the annotation worklist object for a project.

        Args:
            project: The project ID or Project instance.

        Returns:
            The worklist data dict.
        """
        response = self._make_entity_request('GET', project, add_path='worklist')
        return response.json()

    # ------------------------------------------------------------------
    # Annotation statuses
    # ------------------------------------------------------------------

    def get_annotation_statuses(self,
                                project: str | Project,
                                status: str | None = None,
                                user_id: str | None = None,
                                resource_id: str | None = None) -> list[dict]:
        """Get per-resource annotation statuses for a project.

        Args:
            project: The project ID or Project instance.
            status: Optional status filter.
            user_id: Optional user ID filter.
            resource_id: Optional resource ID filter.

        Returns:
            List of annotation status dicts.
        """
        params = {k: v for k, v in {'status': status, 'user_id': user_id,
                                     'resource_id': resource_id}.items() if v is not None}
        response = self._make_entity_request('GET', project, add_path='annotation-statuses',
                                             params=params or None)
        return response.json()

    # ------------------------------------------------------------------
    # Project splits
    # ------------------------------------------------------------------

    def get_splits(
        self,
        project: str | Project,
        split_name: str | None = None,
        as_of_timestamp: str | None = None,
    ) -> list[ProjectResourceSplit]:
        """List resource split assignments for a project.

        Args:
            project: The project ID or Project instance.
            split_name: Optional split name filter.
            as_of_timestamp: Optional historical timestamp filter.

        Returns:
            List of project split assignments.
        """
        params: dict[str, str] = {}
        if split_name is not None:
            params['split_name'] = split_name
        if as_of_timestamp is not None:
            params['as_of_timestamp'] = as_of_timestamp

        response = self._make_entity_request('GET', project, 'splits', params=params or None)
        return [ProjectResourceSplit(**item) for item in response.json()]

    def assign_splits(
        self,
        project: str | Project,
        resources: Sequence[str] | Sequence[Resource],
        split_name: str,
    ) -> None:
        """Assign a split name to multiple project resources.

        Args:
            project: The project ID or Project instance.
            resources: Resources to assign.
            split_name: Split name to assign, such as ``'train'``.
        """

        resources = [self._entid(res) for res in resources]

        self._make_entity_request(
            'POST',
            project,
            'splits',
            json={'resource_ids': list(resources), 'split_name': split_name},
        )

    def get_resource_split(
        self,
        project: str | Project,
        resource: Resource | str,
    ) -> ProjectResourceSplit | None:
        """Get the split assignment for a single resource within a project.

        Returns ``None`` when the resource does not currently have a split
        assignment.
        """
        project_id = self._entid(project)
        resource_id = self._entid(resource)

        try:
            response = self._make_request(
                'GET',
                f'/projects/{project_id}/resources/{resource_id}/split',
            )
        except ItemNotFoundError:
            return None
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                return None
            raise

        data = response.json()
        return ProjectResourceSplit(**data) if data else None

    def reset_annotator_status(self,
                               project: str | Project,
                               resource: str | Resource,
                               annotator: str) -> None:
        """Reset annotation status for a specific annotator on a resource.

        Args:
            project: The project ID or Project instance.
            resource: The resource ID or Resource instance.
            annotator: The annotator's email address.
        """
        proj_id = self._entid(project)
        resource_id = self._entid(resource)
        self._make_request('DELETE',
                           f'/{self.endpoint_base}/{proj_id}/resources/{resource_id}'
                           f'/annotator/{annotator}/status-reset')

    # ------------------------------------------------------------------
    # Download annotations
    # ------------------------------------------------------------------

    def download_annotations(self,
                             project: str | Project,
                             output_path: str | Path,
                             format: str = 'csv',
                             annotators: list[str] | None = None,
                             annotations: list[str] | None = None,
                             from_date: str | None = None,
                             to_date: str | None = None,
                             progress_bar: bool = True) -> None:
        """Download annotation data as a CSV or Excel file.

        Args:
            project: The project ID or Project instance.
            output_path: Local file path to save the downloaded data.
            format: Export format, ``'csv'`` (default) or ``'xlsx'``.
            annotators: Optional list of annotator emails to include.
            annotations: Optional list of annotation identifiers to include.
            from_date: Optional start date filter (ISO string).
            to_date: Optional end date filter (ISO string).
            progress_bar: Whether to display a progress bar.
        """
        from tqdm.auto import tqdm

        proj_id = self._entid(project)
        params: dict = {'format': format}
        if from_date is not None:
            params['from'] = from_date
        if to_date is not None:
            params['to'] = to_date
        if annotators is not None:
            params['annotators[]'] = annotators
        if annotations is not None:
            params['annotations[]'] = annotations

        output_path = Path(output_path)
        with self._stream_entity_request('GET', proj_id,
                                         add_path='download-annotations',
                                         params=params) as response:
            total_size = int(response.headers.get('content-length', 0)) or None
            with tqdm(total=total_size, unit='B', unit_scale=True,
                      disable=not progress_bar) as pbar:
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_bytes(8192):
                        pbar.update(len(chunk))
                        f.write(chunk)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_annotators_stats(self,
                             project: str | Project,
                             email: str | None = None) -> list[dict]:
        """Get per-annotator completion statistics for a project.

        Args:
            project: The project ID or Project instance.
            email: Optional annotator email to filter results.

        Returns:
            List of per-annotator stat dicts.
        """
        params = {'email': email} if email is not None else None
        response = self._make_entity_request('GET', project, add_path='annotators-statistic',
                                             params=params)
        return response.json()

    def get_annotations_stats(self, project: str | Project) -> dict:
        """Get aggregate annotation statistics (counts per type) for a project.

        Args:
            project: The project ID or Project instance.

        Returns:
            Annotation statistics dict.
        """
        response = self._make_entity_request('GET', project, add_path='annotations-statistic')
        return response.json()

    def get_files_matrix_stats(self, project: str | Project) -> dict:
        """Get a matrix of resource × annotator completion statistics.

        Args:
            project: The project ID or Project instance.

        Returns:
            Files-matrix statistics dict.
        """
        response = self._make_entity_request('GET', project, add_path='files-matrix-statistic')
        return response.json()

    def get_annotator_status(self, project: str | Project, email: str) -> dict:
        """Get a specific annotator's progress status in a project.

        Args:
            project: The project ID or Project instance.
            email: The annotator's email address.

        Returns:
            Annotator status dict.
        """
        proj_id = self._entid(project)
        response = self._make_request('GET',
                                      f'/{self.endpoint_base}/{proj_id}/users/{email}/status')
        return response.json()

    # ------------------------------------------------------------------
    # Review messages
    # ------------------------------------------------------------------

    def get_review_messages(self,
                            project: str | Project,
                            annotator: str | None = None,
                            resource_id: str | None = None,
                            statuses: list[str] | None = None) -> list[dict]:
        """Get review feedback messages for a project.

        Args:
            project: The project ID or Project instance.
            annotator: Optional annotator email filter.
            resource_id: Optional resource ID filter.
            statuses: Optional list of status strings to filter by.

        Returns:
            List of review message dicts.
        """
        params: dict = {}
        if annotator is not None:
            params['annotator'] = annotator
        if resource_id is not None:
            params['resourceId'] = resource_id
        if statuses is not None:
            params['statuses'] = statuses
        response = self._make_entity_request('GET', project, add_path='reviewmessages',
                                             params=params or None)
        return response.json()

    # ------------------------------------------------------------------
    # Project models
    # ------------------------------------------------------------------

    def get_models(self, project: str | Project) -> list[dict]:
        """List ML models associated with a project.

        Args:
            project: The project ID or Project instance.

        Returns:
            List of model dicts.
        """
        response = self._make_entity_request('GET', project, add_path='models')
        return response.json()
