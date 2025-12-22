"""Project entity module for DataMint API."""
from datetime import datetime
import logging
from typing import Sequence, Literal, TYPE_CHECKING
from .base_entity import BaseEntity, MISSING_FIELD
from typing import Any
import webbrowser
from pydantic import PrivateAttr

if TYPE_CHECKING:
    from datamint.api.endpoints.projects_api import ProjectsApi
    from .resource import Resource

logger = logging.getLogger(__name__)


class Project(BaseEntity):
    """Pydantic Model representing a DataMint project.

    This class models a project entity from the DataMint API, containing
    information about the project, its dataset, worklist, AI model, and
    annotation statistics.

    Attributes:
        id: Unique identifier for the project
        name: Human-readable name of the project
        description: Optional description of the project
        created_at: ISO timestamp when the project was created
        created_by: Email of the user who created the project
        dataset_id: ID of the associated dataset
        worklist_id: ID of the associated worklist
        ai_model_id: Optional ID of the associated AI model
        viewable_ai_segs: Optional configuration for viewable AI segments
        editable_ai_segs: Optional configuration for editable AI segments
        archived: Whether the project is archived
        resource_count: Total number of resources in the project
        annotated_resource_count: Number of resources that have been annotated
        most_recent_experiment: Optional information about the most recent experiment
        closed_resources_count: Number of resources marked as closed/completed
        resources_to_annotate_count: Number of resources still needing annotation
        annotators: List of annotators assigned to this project
    """
    id: str
    name: str
    created_at: str
    created_by: str
    dataset_id: str
    worklist_id: str
    archived: bool
    resource_count: int
    annotated_resource_count: int
    description: str | None
    viewable_ai_segs: list | None
    editable_ai_segs: list | None
    registered_model: Any | None = MISSING_FIELD
    ai_model_id: str | None = MISSING_FIELD
    closed_resources_count: int = MISSING_FIELD
    resources_to_annotate_count: int = MISSING_FIELD
    most_recent_experiment: str | None = MISSING_FIELD
    annotators: list[dict] = MISSING_FIELD
    archived_on: str | None = MISSING_FIELD
    archived_by: str | None = MISSING_FIELD
    is_active_learning: bool = MISSING_FIELD
    two_up_display: bool = MISSING_FIELD
    require_review: bool = MISSING_FIELD

    _api: 'ProjectsApi' = PrivateAttr()

    def fetch_resources(self) -> Sequence['Resource']:
        """Fetch resources associated with this project from the API,
        IMPORTANT: It always fetches fresh data from the server.

        Returns:
            List of Resource instances associated with the project.
        """
        return self._api.get_project_resources(self.id)

    def download_resources_datas(self, progress_bar: bool = True) -> None:
        """Downloads all project resources in parallel for faster subsequent access.

        This method downloads and caches all resource file data concurrently,
        skipping resources that are already cached. This dramatically improves
        performance when working with large projects.

        Args:
            progress_bar: Whether to show a progress bar. Default is True.

        Example:
            >>> proj = api.projects.get_by_name("My Project")
            >>> proj.download_resources()  # Cache all resources in parallel
            >>> # Now fetch_file_data() will be instantaneous for cached resources
            >>> for res in proj.fetch_resources():
            ...     data = res.fetch_file_data(use_cache=True)
        """
        return self.cache_resources(progress_bar=progress_bar)

    def cache_resources(self, progress_bar: bool = True) -> None:
        """Cache all project resources in parallel for faster subsequent access.

        This method downloads and caches all resource file data concurrently,
        skipping resources that are already cached. This dramatically improves
        performance when working with large projects.

        Args:
            progress_bar: Whether to show a progress bar. Default is True.

        Example:
            >>> proj = api.projects.get_by_name("My Project")
            >>> proj.cache_resources()  # Cache all resources in parallel
            >>> # Now fetch_file_data() will be instantaneous for cached resources
            >>> for res in proj.fetch_resources():
            ...     data = res.fetch_file_data(use_cache=True)
        """
        resources = self.fetch_resources()
        self._api.resources_api.cache_resources(resources, progress_bar=progress_bar)

    def set_work_status(self, resource: 'Resource', status: Literal['opened', 'annotated', 'closed']) -> None:
        """Set the status of a resource.

        Args:
            resource: The resource unique id or a resource object.
            status: The new status to set.
        """

        return self._api.set_work_status(self, resource, status)

    @property
    def url(self) -> str:
        """Get the URL to access this project in the DataMint web application."""
        base_url = self._api.config.web_app_url
        return f'{base_url}/projects/edit/{self.id}'

    def show(self) -> None:
        """Open the project in the default web browser."""
        webbrowser.open(self.url)

    def as_torch_dataset(self,
                         root_dir: str | None = None,
                         auto_update: bool = True,
                         return_as_semantic_segmentation: bool = False):
        from datamint.dataset import Dataset
        return Dataset(project_name=self.name,
                       root=root_dir,
                       auto_update=auto_update,
                       return_as_semantic_segmentation=return_as_semantic_segmentation,
                       all_annotations=True)
