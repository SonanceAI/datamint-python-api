from .base_api import ApiConfig, BaseApi
from .endpoints import (ProjectsApi, ResourcesApi, AnnotationsApi,
                        ChannelsApi, UsersApi, DatasetsInfoApi,
                        AnnotationSetsApi, DeployModelApi
                        )
from .endpoints.models_api import ModelsApi
import datamint.configs
from datamint.exceptions import DatamintException


class Api:
    """Main API client that provides access to all endpoint handlers."""
    DEFAULT_SERVER_URL = 'https://api.datamint.io'
    DATAMINT_API_VENV_NAME = datamint.configs.ENV_VARS[datamint.configs.APIKEY_KEY]

    _API_MAP: dict[str, type[BaseApi]] = {
        'projects': ProjectsApi,
        'resources': ResourcesApi,
        'annotations': AnnotationsApi,
        'channels': ChannelsApi,
        'users': UsersApi,
        'datasets': DatasetsInfoApi,
        'models': ModelsApi,
        'annotationsets': AnnotationSetsApi,
        'deploy': DeployModelApi,
    }

    def __init__(self,
                 server_url: str | None = None,
                 api_key: str | None = None,
                 timeout: float = 60.0, max_retries: int = 2,
                 check_connection: bool = True,
                 verify_ssl: bool | str = True) -> None:
        """Initialize the API client.

        Args:
            server_url: Base URL for the API
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            check_connection: Whether to check connection on initialization
            verify_ssl: Whether to verify SSL certificates. Default is True.
                Set to False only in development environments with self-signed certificates.
                Can also be a path to a CA bundle file for custom certificate verification.
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
            max_retries=max_retries,
            verify_ssl=verify_ssl
        )
        self.high_config = ApiConfig(
            server_url=server_url,
            api_key=api_key,
            timeout=timeout*5,
            max_retries=max_retries,
            verify_ssl=verify_ssl,
        )
        self.mlflow_config = ApiConfig(
            server_url=server_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            port=5000,
            verify_ssl=verify_ssl
        )
        self._client = None
        self._highclient = None
        self._mlclient = None
        self._endpoints: dict[str, BaseApi] = {}
        if check_connection:
            self.check_connection()

    def check_connection(self):
        try:
            self.projects.get_list(limit=1)
        except Exception as e:
            raise DatamintException("Error connecting to the Datamint API." +
                                    f" Please check your api_key and/or other configurations.") from e

    def _get_endpoint(self, name: str, is_mlflow: bool = False):
        if is_mlflow:
            if self._mlclient is None:
                self._mlclient = BaseApi._create_client(self.mlflow_config)
            client = self._mlclient
        elif name in ['resources', 'annotations']:
            if self._highclient is None:
                self._highclient = BaseApi._create_client(self.high_config)
            client = self._highclient
        else:
            if self._client is None:
                self._client = BaseApi._create_client(self.config)
            client = self._client
        if name not in self._endpoints:
            api_class = self._API_MAP[name]
            endpoint = api_class(self.config, client)
            # Inject this API instance into the endpoint so it can inject into entities
            endpoint._api_instance = self
            self._endpoints[name] = endpoint
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

    @property
    def models(self) -> ModelsApi:
        return self._get_endpoint('models')

    @property
    def annotationsets(self) -> AnnotationSetsApi:
        return self._get_endpoint('annotationsets')

    @property
    def deploy(self) -> DeployModelApi:
        """Access deployment management endpoints."""
        return self._get_endpoint('deploy', is_mlflow=True)
