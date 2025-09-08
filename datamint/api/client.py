from typing import Optional
import httpx
from .base_api import ApiConfig
from .endpoints import ProjectsApi, ResourcesApi


class Api:
    """Main API client that provides access to all endpoint handlers."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 timeout: float = 30.0, max_retries: int = 3,
                 client: Optional[httpx.Client] = None) -> None:
        """Initialize the API client.
        
        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            client: Optional HTTP client instance
        """
        self.config = ApiConfig(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        self._client = client
        
        # Initialize endpoint handlers
        self._projects = None
        self._annotations = None
        self._resources = None
    
    @property
    def projects(self) -> ProjectsApi:
        """Access to project-related endpoints."""
        if self._projects is None:
            self._projects = ProjectsApi(self.config, self._client)
        return self._projects
    
    @property
    def resources(self) -> ResourcesApi:
        """Access to resource-related endpoints."""
        if self._resources is None:
            self._resources = ResourcesApi(self.config, self._client)
        return self._resources
    
    # @property
    # def annotations(self) -> AnnotationsApi:
    #     """Access to annotation-related endpoints."""
    #     if self._annotations is None:
    #         self._annotations = AnnotationsApi(self.config, self._client)
    #     return self._annotations
    
    # def close(self) -> None:
    #     """Close the HTTP client connections."""
    #     if self._projects and self._projects.client:
    #         self._projects.client.close()
    #     if self._annotations and self._annotations.client:
    #         self._annotations.client.close()
    
    # def __enter__(self):
    #     """Context manager entry."""
    #     return self
    
    # def __exit__(self, exc_type, exc_val, exc_tb):
    #     """Context manager exit."""
    #     self.close()
