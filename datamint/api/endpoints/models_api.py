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
    from mlflow.models import ModelInputExample, ModelSignature

    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities.annotations.annotation_spec import AnnotationSpec
    from datamint.entities.project import Project
    from datamint.mlflow.flavors.model import BaseDatamintModel
    from datamint.mlflow.flavors.task_type import TaskType

    from .projects_api import ProjectsApi


class ModelsApi(BaseApi):
    """API handler for the model registry.

    Wraps MLflow's model registry (registered models / model versions) behind
    plain Python objects (:class:`~.model_types.Model`, :class:`~.model_types.ModelVersion`)
    so callers never need to know MLflow's object model.
    """

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None,
                 deploy_api: DeployModelApi | None = None,
                 projects_api: 'ProjectsApi | None' = None) -> None:
        super().__init__(config, client)
        self._deploy_api = deploy_api or DeployModelApi(config, client=client)
        self._projects_api = projects_api

    @property
    def _mlflow_client(self) -> mlflow.tracking.MlflowClient:
        return mlflow.tracking.MlflowClient()

    @property
    def projects_api(self) -> 'ProjectsApi':
        if self._projects_api is None:
            from .projects_api import ProjectsApi
            self._projects_api = ProjectsApi(self.config, client=self.client)
        return self._projects_api

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

    def get_projects(self, model_name: str, customer_id: str | None = None) -> list['Project']:
        """Get all projects a registered model is associated with.

        Args:
            model_name: Name of the registered model.
            customer_id: Optional customer ID to scope the lookup.
        """
        payload = {'model_name': model_name}
        if customer_id is not None:
            payload['customer_id'] = customer_id
        response = self._make_request('POST', 'datamint/api/v1/model-info/get-project', json=payload)
        project_ids = response.json().get('project_ids', [])
        return [self.projects_api.get_by_id(pid) for pid in project_ids]

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

    def log_model(self,
                  datamint_model: 'BaseDatamintModel',
                  project: 'str | Project',
                  model_name: str,
                  *,
                  dataset: 'DatamintBaseDataset | None' = None,
                  task_type: 'TaskType | str | None' = None,
                  annotation_specs: 'Sequence[AnnotationSpec] | None' = None,
                  supported_modes: 'Sequence[str] | None' = None,
                  code_paths: 'Sequence[str] | None' = None,
                  artifacts: dict | None = None,
                  signature: 'ModelSignature | None' = None,
                  input_example: 'ModelInputExample | None' = None,
                  pip_requirements: 'Sequence[str] | None' = None,
                  extra_pip_requirements: 'Sequence[str] | None' = None,
                  ) -> Model:
        """Log a model to the Datamint model registry, opening its own MLflow run.

        A thin wrapper around :func:`~datamint.mlflow.flavors.datamint_flavor.log_model` 
        for models trained outside a Datamint trainer (the trainers log their models
        automatically). It resolves the active project, starts an MLflow run,
        and registers the model under ``model_name`` -- same run-handling as
        :meth:`clone_model`.

        Args:
            datamint_model: The trained model to log.
            project: Project (name, ID, or :class:`~datamint.entities.project.Project`)
                to register the model under.
            model_name: Name to register the model under. Also used as the
                MLflow experiment name.
            dataset: The dataset the model was trained on, used to derive
                ``annotation_specs`` automatically when they aren't given
                explicitly. Same dataset object you already have from training
                (e.g. an :class:`~datamint.dataset.ImageDataset`).
            task_type: Task type used to pick the right annotation-spec
                builder. Defaults to ``datamint_model.task_type`` when omitted.
            annotation_specs: Explicit annotation specs. Takes precedence over
                anything derived from ``dataset``.
            supported_modes: Prediction modes the model supports.
            code_paths: Local paths to custom code files/dirs the model's
                class depends on.
            artifacts: Extra artifacts to bundle with the model.
            signature: MLflow model signature.
            input_example: MLflow input example.
            pip_requirements: Exact pip requirements for the model environment.
            extra_pip_requirements: Additional pip requirements on top of the
                inferred defaults.

        Returns:
            The newly registered :class:`~.model_types.Model`.
        """
        from datamint.mlflow.flavors import datamint_flavor
        from datamint.mlflow.flavors.annotation_specs import build_annotation_specs_for_task
        from datamint.mlflow.tracking.fluent import (
            _reset_active_project,
            get_active_project_id,
            set_project,
        )

        resolved_task_type = task_type or getattr(datamint_model, 'task_type', None)

        resolved_specs = annotation_specs
        if resolved_specs is None and dataset is not None:
            resolved_specs = build_annotation_specs_for_task(resolved_task_type, dataset)

        previous_project_id = get_active_project_id()
        try:
            set_project(project)
            mlflow.set_experiment(model_name)
            with mlflow.start_run(run_name=f"log_{model_name}"):
                datamint_flavor.log_model(
                    datamint_model,
                    task_type=resolved_task_type,
                    supported_modes=supported_modes,
                    annotation_specs=resolved_specs,
                    model_name=model_name,
                    code_paths=code_paths,
                    artifacts=artifacts,
                    signature=signature,
                    input_example=input_example,
                    pip_requirements=pip_requirements,
                    extra_pip_requirements=extra_pip_requirements,
                )
            return self.get_by_name(model_name)
        finally:
            if previous_project_id is not None:
                set_project(previous_project_id)
            else:
                _reset_active_project()
