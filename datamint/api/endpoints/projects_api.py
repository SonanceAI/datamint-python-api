from typing import Sequence, Literal, TYPE_CHECKING, overload
from ..entity_base_api import ApiConfig, CRUDEntityApi
from datamint.entities.project import Project
import httpx
from datamint.entities.resource import Resource
if TYPE_CHECKING:
    from .resources_api import ResourcesApi


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
        from .resources_api import ResourcesApi
        super().__init__(config, Project, 'projects', client)
        self.resources_api = resources_api or ResourcesApi(config, client, projects_api=self)

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
               *,
               return_entity: Literal[True] = True
               ) -> Project: ...

    @overload
    def create(self,
               name: str,
               description: str,
               resources_ids: list[str] | None = None,
               is_active_learning: bool = False,
               two_up_display: bool = False,
               *,
               return_entity: Literal[False]
               ) -> str: ...

    def create(self,
               name: str,
               description: str,
               resources_ids: list[str] | None = None,
               is_active_learning: bool = False,
               two_up_display: bool = False,
               *,
               return_entity: bool = True
               ) -> str | Project:
        """Create a new project.

        Args:
            name: The name of the project.
            description: The description of the project.
            resources_ids: The list of resource ids to be included in the project.
            is_active_learning: Whether the project is an active learning project or not.
            two_up_display: Allow annotators to display multiple resources for annotation.
            return_entity: Whether to return the created Project instance or just its ID.

        Returns:
            The id of the created project.
        """
        resources_ids = resources_ids or []
        project_data = {'name': name,
                        'is_active_learning': is_active_learning,
                        'resource_ids': resources_ids,
                        'annotation_set': {
                            "annotators": [],
                            "resource_ids": resources_ids,
                            "annotations": [],
                            "frame_labels": [],
                            "image_labels": [],
                        },
                        "two_up_display": two_up_display,
                        "require_review": False,
                        'description': description}

        return self._create(project_data, return_entity=return_entity)

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
