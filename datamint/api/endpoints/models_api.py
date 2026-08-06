"""API handler for the model registry, backed by MLflow."""
from collections.abc import Sequence
from typing import TYPE_CHECKING

import httpx
import mlflow.exceptions
import mlflow.tracking

from datamint.exceptions import ItemNotFoundError

from ..entity_base_api import ApiConfig, BaseApi
from .deploy_model_api import DeployModelApi
from .model_types import Model, ModelVersion

if TYPE_CHECKING:
    from datamint.entities.project import Project


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

    def delete_model_version(self, name: str, version: str | int) -> None:
        """Delete a single model version.

        Args:
            name: Name of the registered model.
            version: Version number to delete.
        """
        self._mlflow_client.delete_model_version(name, str(version))

    def delete_registered_model(self, name: str) -> None:
        """Delete a registered model and all of its versions.

        Args:
            name: Name of the registered model to delete.
        """
        self._mlflow_client.delete_registered_model(name)

    def _resolve_version(self,
                         model: 'str | Model | ModelVersion',
                         version: str | int | None,
                         alias: str | None) -> ModelVersion:
        if isinstance(model, ModelVersion):
            if version is not None or alias is not None:
                raise TypeError("'version'/'alias' must not be passed when 'model' is already a ModelVersion.")
            return model

        if (version is None) == (alias is None):
            raise TypeError("clone_model() requires exactly one of 'version' or 'alias'.")

        if isinstance(model, Model):
            registered_model = model
        else:
            registered_model = self.get_by_name(model)
            if registered_model is None:
                raise ItemNotFoundError('Model', {'name': model})

        if alias is not None:
            resolved = registered_model.get_latest_version(alias=alias)
            if resolved is None:
                raise ItemNotFoundError('ModelVersion', {'name': registered_model.name, 'alias': alias})
            return resolved

        for v in registered_model.get_versions():
            if str(v.version) == str(version):
                return v
        raise ItemNotFoundError('ModelVersion', {'name': registered_model.name, 'version': version})

    def clone_model(self,
                    model: 'str | Model | ModelVersion',
                    target_project: 'str | Project',
                    *,
                    version: str | int | None = None,
                    alias: str | None = None,
                    target_model_name: str | None = None,
                    code_paths: 'Sequence[str] | None' = None) -> Model:
        """Clone a model version from the active project into another project.

        The source version is resolved against whichever project is currently
        active (see :func:`datamint.mlflow.set_project`) -- same as every other
        ``ModelsApi`` lookup. Only models logged with the ``datamint`` MLflow
        flavor are supported, since that's what carries the task type, supported
        modes, and annotation specs this method copies over.

        Args:
            model: Registered model name, :class:`~.model_types.Model`, or a specific
                :class:`~.model_types.ModelVersion` to clone. When a name or ``Model``
                is passed, exactly one of ``version``/``alias`` must also be given.
            target_project: Project (name, ID, or :class:`~datamint.entities.project.Project`)
                to register the cloned model under.
            version: Version number to clone. Mutually exclusive with ``alias``.
            alias: Alias to clone (e.g. ``"champion"``). Mutually exclusive with ``version``.
            target_model_name: Name to register the clone under in ``target_project``.
                Defaults to the source model's name.
            code_paths: Local paths to custom code files/dirs the model's class depends
                on (same meaning as ``datamint_flavor.log_model``'s ``code_paths``).

        Returns:
            The newly registered :class:`~.model_types.Model` in ``target_project``.

        Raises:
            ValueError: If the source version wasn't logged with the ``datamint`` flavor.
        """
       
        from datamint.mlflow.flavors import datamint_flavor
        from datamint.mlflow.flavors.datamint_flavor import FLAVOR_NAME
        from datamint.mlflow.tracking.fluent import (
            _reset_active_project,
            get_active_project_id,
            set_project,
        )

        source_version = self._resolve_version(model, version, alias)

        model_info = mlflow.models.get_model_info(source_version.source)
        if FLAVOR_NAME not in model_info.flavors:
            raise ValueError(
                f"Model {source_version.name!r} version {source_version.version!r} was not logged "
                f"with the 'datamint' MLflow flavor; clone_model() only supports datamint-flavor models."
            )

        task_type = source_version.get_task_type()
        supported_modes = source_version.get_supported_modes()
        annotation_specs = source_version.get_annotation_specs()
        loaded_model = datamint_flavor.load_model(source_version.source)

        target_name = target_model_name or source_version.name

        previous_project_id = get_active_project_id()
        try:
            set_project(target_project)
            mlflow.set_experiment(target_name)
            with mlflow.start_run(run_name=f"clone_{source_version.name}_v{source_version.version}"):
                datamint_flavor.log_model(
                    loaded_model,
                    task_type=task_type,
                    supported_modes=supported_modes,
                    annotation_specs=annotation_specs,
                    model_name=target_name,
                    code_paths=code_paths,
                )
            return self.get_by_name(target_name)
        finally:
            if previous_project_id is not None:
                set_project(previous_project_id)
            else:
                _reset_active_project()
