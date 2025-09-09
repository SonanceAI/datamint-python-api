from typing import Optional
import httpx
from .base_api import ApiConfig
from .endpoints import ProjectsApi, ResourcesApi, AnnotationsApi, ChannelsApi
import datamint.configs
from datamint.exceptions import DatamintException
import asyncio


class Api:
    """Main API client that provides access to all endpoint handlers."""
    DEFAULT_SERVER_URL = 'https://api.datamint.io'
    DATAMINT_API_VENV_NAME = datamint.configs.ENV_VARS[datamint.configs.APIKEY_KEY]

    def __init__(self,
                 server_url: str | None = None,
                 api_key: Optional[str] = None,
                 timeout: float = 30.0, max_retries: int = 3,
                 check_connection: bool = True) -> None:
        """Initialize the API client.

        Args:
            base_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            client: Optional HTTP client instance
        """
        if server_url is None:
            server_url = datamint.configs.get_value(datamint.configs.APIURL_KEY)
            if server_url is None:
                server_url = Api.DEFAULT_SERVER_URL
        server_url = server_url.rstrip('/')
        if api_key is None:
            api_key = datamint.configs.get_value(datamint.configs.APIKEY_KEY)
            if api_key is None:
                msg = f"API key not provided! Use the environment variable " + \
                    f"{Api.DATAMINT_API_VENV_NAME} or pass it as an argument."
                raise DatamintException(msg)
        # self.semaphore = asyncio.Semaphore(20)

        self.config = ApiConfig(
            server_url=server_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        self._client = None
        # Initialize endpoint handlers
        self._projects = None
        self._annotations = None
        self._resources = None
        self._channels = None

        if check_connection:
            self.check_connection()

    def check_connection(self):
        try:
            self.projects.get_list(limit=1)
        except Exception as e:
            raise DatamintException("Error connecting to the Datamint API." +
                                    f" Please check your api_key and/or other configurations. {e}")

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

    @property
    def annotations(self) -> AnnotationsApi:
        """Access to annotation-related endpoints."""
        if self._annotations is None:
            self._annotations = AnnotationsApi(self.config, self._client)
        return self._annotations

    @property
    def channels(self) -> ChannelsApi:
        """Access to channel-related endpoints."""
        if self._channels is None:
            self._channels = ChannelsApi(self.config, self._client)
        return self._channels

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
