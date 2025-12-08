"""Dataset entity module for DataMint API."""

import logging
from typing import TYPE_CHECKING, Sequence
from pydantic import PrivateAttr

from .base_entity import BaseEntity, MISSING_FIELD

if TYPE_CHECKING:
    from datamint.api.client import Api
    from .resource import Resource
    from .project import Project
    from datamint.api.endpoints.datasetsinfo_api import DatasetsInfoApi

logger = logging.getLogger(__name__)


class DatasetInfo(BaseEntity):
    """Pydantic Model representing a DataMint dataset.
    
    This class provides access to dataset information and related entities
    like resources and projects.
    """

    id: str
    name: str
    created_at: str  # ISO timestamp string
    created_by: str
    description: str
    customer_id: str
    updated_at: str | None
    total_resource: int
    resource_ids: list[str]

    _api: 'DatasetsInfoApi' = PrivateAttr()

    def __init__(self, **data):
        """Initialize the dataset info entity."""
        super().__init__(**data)
        
        # Cache for lazy-loaded data
        # self._resources_cache: Sequence['Resource'] | None = None
        # self._projects_cache: Sequence['Project'] | None = None
    
    # def get_resources(
    #     self,
    #     refresh: bool = False,
    #     limit: int | None = None
    # ) -> Sequence['Resource']:
    #     """Get all resources in this dataset.
        
    #     Results are cached after the first call unless refresh=True.
        
    #     Args:
    #         api: Optional API client. Uses the one from set_api() if not provided.
    #         refresh: If True, bypass cache and fetch fresh data
            
    #     Returns:
    #         List of Resource instances in this dataset
            
    #     Raises:
    #         RuntimeError: If no API client is available
            
    #     Example:
    #         >>> dataset = api._datasetsinfo.get_by_id("dataset-id")
    #         >>> dataset.set_api(api)
    #         >>> resources = dataset.get_resources()
    #     """
    #     if refresh or self._resources_cache is None:
    #         # Fetch resources by their IDs
    #         resources = []
    #         for resource_id in self.resource_ids:
    #             try:
    #                 resource = self._api.get.get_by_id(resource_id)
    #                 resource.set_api(self._api)
    #                 resources.append(resource)
    #             except Exception as e:
    #                 logger.warning(f"Failed to fetch resource {resource_id}: {e}")
            
    #         self._resources_cache = resources
        
    #     return self._resources_cache

    # def get_projects(
    #     self,
    #     api: 'Api | None' = None,
    #     refresh: bool = False
    # ) -> Sequence['Project']:
    #     """Get all projects associated with this dataset.
        
    #     Results are cached after the first call unless refresh=True.
        
    #     Args:
    #         refresh: If True, bypass cache and fetch fresh data
            
    #     Returns:
    #         List of Project instances
            
    #     Raises:
    #         RuntimeError: If no API client is available
            
    #     Example:
    #         >>> dataset = api.datasetsinfo.get_by_id("dataset-id")
    #         >>> projects = dataset.get_projects()
    #     """
    #     if refresh or self._projects_cache is None:
            
    #         # Get all projects and filter by dataset_id
    #         all_projects = api_client.projects.get_all()
    #         projects = [p for p in all_projects if p.dataset_id == self.id]
            
    #         self._projects_cache = projects
        
    #     return self._projects_cache
    
    def invalidate_cache(self) -> None:
        """Invalidate all cached relationship data.
        
        This forces fresh data fetches on the next access.
        """
        self._resources_cache = None
        self._projects_cache = None
        logger.debug(f"Invalidated cache for dataset {self.id}")

