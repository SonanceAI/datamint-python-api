from typing import Optional
import httpx
from .base_api import ApiConfig
from .endpoints import ProjectsApi, ResourcesApi, AnnotationsApi, ChannelsApi, UsersApi, DatasetsInfoApi
import datamint.configs
from datamint.exceptions import DatamintException
import asyncio


class Api:
    """Main API client that provides access to all endpoint handlers."""
    DEFAULT_SERVER_URL = 'https://api.datamint.io'
    DATAMINT_API_VENV_NAME = datamint.configs.ENV_VARS[datamint.configs.APIKEY_KEY]

    _API_MAP = {
        'projects': ProjectsApi,
        'resources': ResourcesApi,
        'annotations': AnnotationsApi,
        'channels': ChannelsApi,
        'users': UsersApi,
        'datasets': DatasetsInfoApi
    }

    def __init__(self,
                 server_url: str | None = None,
                 api_key: Optional[str] = None,
                 timeout: float = 60.0, max_retries: int = 2,
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
        self.config = ApiConfig(
            server_url=server_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries
        )
        self._client = None
        self._endpoints = {}
        if check_connection:
            self.check_connection()

    def check_connection(self):
        try:
            self.projects.get_list(limit=1)
        except Exception as e:
            raise DatamintException("Error connecting to the Datamint API." +
                                    f" Please check your api_key and/or other configurations. {e}")

    def _get_endpoint(self, name: str):
        if name not in self._endpoints:
            api_class = self._API_MAP[name]
            self._endpoints[name] = api_class(self.config, self._client)
        return self._endpoints[name]

    @property
    def projects(self) -> ProjectsApi:
        return self._get_endpoint('projects')
    @property
    def resources(self) -> ResourcesApi:
        return self._get_endpoint('resources')
    @property
    def annotations(self) -> AnnotationsApi:
        return self._get_endpoint('annotations')
    @property
    def channels(self) -> ChannelsApi:
        return self._get_endpoint('channels')
    @property
    def users(self) -> UsersApi:
        return self._get_endpoint('users')
    @property
    def _datasetsinfo(self) -> DatasetsInfoApi:
        """Internal property to access DatasetsInfoApi."""
        return self._get_endpoint('datasets')