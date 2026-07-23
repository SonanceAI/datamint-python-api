"""API handler for the model registry, backed by MLflow."""
import httpx
import mlflow.exceptions
import mlflow.tracking

from ..entity_base_api import ApiConfig, BaseApi
from .deploy_model_api import DeployModelApi
from .model_types import Model


class ModelsApi(BaseApi):
    """API handler for the model registry.

    Wraps MLflow's model registry (registered models / model versions) behind
    plain Python objects (:class:`~.model_types.Model`, :class:`~.model_types.ModelVersion`)
    so callers never need to know MLflow's object model.
    """

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None,
                 deploy_api: DeployModelApi | None = None) -> None:
        super().__init__(config, client)
        self._deploy_api = deploy_api or DeployModelApi(config, client=client)

    @property
    def _mlflow_client(self) -> mlflow.tracking.MlflowClient:
        import datamint.mlflow 
        return mlflow.tracking.MlflowClient()

    def get_list(self,
                only_deployed: bool = False,
                max_results: int | None = None) -> list[Model]:
        """List registered models.

        Args:
            only_deployed: If ``True``, only return models with a deployed image.
            max_results: Maximum number of models to return. If ``None``, all
                registered models are returned (paginating through the registry).
        """
        if max_results is not None:
            raw_models = list(self._mlflow_client.search_registered_models(max_results=max_results))
        else:
            raw_models = []
            page_token = None
            while True:
                page = self._mlflow_client.search_registered_models(page_token=page_token)
                raw_models.extend(page)
                page_token = page.token
                if not page_token:
                    break

        models = [Model(_raw=m, _api=self) for m in raw_models]
        if only_deployed:
            models = [m for m in models if m.is_deployed()]
        return models

    def get_all(self, only_deployed: bool = False, max_results: int | None = None) -> list[Model]:
        """Alias for :meth:`get_list`, kept for backwards compatibility with existing call sites."""
        return self.get_list(only_deployed=only_deployed, max_results=max_results)

    def get_by_name(self, name: str) -> Model | None:
        """Get a registered model by name, or ``None`` if it does not exist."""
        try:
            raw_model = self._mlflow_client.get_registered_model(name)
        except mlflow.exceptions.MlflowException as e:
            if e.error_code == 'RESOURCE_DOES_NOT_EXIST':
                return None
            raise
        return Model(_raw=raw_model, _api=self)

    def create(self, name: str, description: str | None = None, exists_ok: bool = True) -> Model:
        """Create a new registered model.

        Args:
            name: Name of the model to register.
            description: Optional description.
            exists_ok: If ``True`` (default), return the existing model instead of
                raising when a model with this name already exists.
        """
        try:
            raw_model = self._mlflow_client.create_registered_model(name, description=description)
        except mlflow.exceptions.MlflowException as e:
            if exists_ok and e.error_code == 'RESOURCE_ALREADY_EXISTS':
                return self.get_by_name(name)
            raise
        return Model(_raw=raw_model, _api=self)
