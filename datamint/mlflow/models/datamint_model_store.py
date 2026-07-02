from mlflow.store.model_registry.rest_store import RestStore
from datamint.mlflow.store_utils import _resolve_project_id, _inject_project_id_into_body
from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import message_to_json
from typing_extensions import override
from mlflow.entities.model_registry import ModelVersion, RegisteredModel
from functools import partial

from mlflow.protos.model_registry_pb2 import (
    CreateModelVersion,
    CreateRegisteredModel,
    DeleteModelVersion,
    DeleteModelVersionTag,
    DeleteRegisteredModel,
    DeleteRegisteredModelAlias,
    DeleteRegisteredModelTag,
    GetLatestVersions,
    GetModelVersion,
    GetModelVersionByAlias,
    GetModelVersionDownloadUri,
    GetRegisteredModel,
    ModelRegistryService,
    RenameRegisteredModel,
    SearchModelVersions,
    SearchRegisteredModels,
    SetModelVersionTag,
    SetRegisteredModelAlias,
    SetRegisteredModelTag,
    TransitionModelVersionStage,
    UpdateModelVersion,
    UpdateRegisteredModel,
)


class DatamintModelRegistryStore(RestStore):
    """
    A model registry store that integrates with the Datamint platform.

    When connected to a Datamint server (detected via URI), the store automatically
    injects project IDs into requests. For standard MLflow stores, it falls back
    to the parent ``RestStore`` behavior.
    """

    def __init__(self, store_uri: str, artifact_uri=None, force_valid=True):
        # Ensure MLflow environment is configured when store is initialized
        from datamint.mlflow.env_utils import setup_mlflow_environment
        from mlflow.utils.credentials import get_default_host_creds

        setup_mlflow_environment()

        if store_uri.startswith('datamint://') or 'datamint.io' in store_uri or force_valid:
            self._is_datamint = True
        else:
            self._is_datamint = False

        store_uri_clean = store_uri.split('datamint://', maxsplit=1)[-1]
        get_host_creds = partial(get_default_host_creds, store_uri_clean)
        super().__init__(get_host_creds=get_host_creds)

    # -- Helpers -------------------------------------------------------------

    def _should_use_original(self) -> bool:
        """Return ``True`` when connected to a non-Datamint (plain MLflow) backend."""
        return not self._is_datamint

    @override
    def get_registered_model(self, name, project_id: str | None = None):
        """
        Get registered model instance by name.

        Args:
            name: Registered model name.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        if self._should_use_original():
            return super().get_registered_model(name)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(GetRegisteredModel(name=name))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(GetRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    @override
    def create_registered_model(self, name, tags=None, description=None, deployment_job_id=None, project_id: str | None = None):
        """
        Create a new registered model in backend store.

        Args:
            name: Name of the new model. This is expected to be unique in the backend store.
            tags: A list of :py:class:`mlflow.entities.model_registry.RegisteredModelTag`
                instances associated with this registered model.
            description: Description of the model.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.RegisteredModel`
            created in the backend.
        """
        if self._should_use_original():
            return super().create_registered_model(name, tags, description, deployment_job_id)

        resolved_project_id = _resolve_project_id(project_id)
        proto_tags = [tag.to_proto() for tag in tags or []]
        req_body = message_to_json(
            CreateRegisteredModel(name=name, tags=proto_tags, description=description)
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(CreateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    @override
    def update_registered_model(self, name, description, deployment_job_id=None, project_id: str | None = None):
        """
        Update description of the registered model.

        Args:
            name: Registered model name.
            description: New description.
            deployment_job_id: Optional deployment job ID.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.
        """
        if self._should_use_original():
            return super().update_registered_model(name, description, deployment_job_id)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(UpdateRegisteredModel(name=name, description=description))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(UpdateRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    @override
    def rename_registered_model(self, name, new_name, project_id: str | None = None):
        """
        Rename the registered model.

        Args:
            name: Registered model name.
            new_name: New proposed name.

        Returns:
            A single updated :py:class:`mlflow.entities.model_registry.RegisteredModel` object.

        """
        if self._should_use_original():
            return super().rename_registered_model(name, new_name)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(RenameRegisteredModel(name=name, new_name=new_name))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(RenameRegisteredModel, req_body)
        return RegisteredModel.from_proto(response_proto.registered_model)

    @override
    def delete_registered_model(self, name, project_id: str | None = None):
        """
        Delete the registered model.
        Backend raises exception if a registered model with given name does not exist.

        Args:
            name: Registered model name.

        Returns:
            None
        """
        if self._should_use_original():
            super().delete_registered_model(name)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(DeleteRegisteredModel(name=name))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(DeleteRegisteredModel, req_body)

    @override
    def search_registered_models(self, filter_string=None, max_results=None, order_by=None, page_token=None, project_id: str | None = None):
        """
        Search for registered models in backend that satisfy the filter criteria.

        Args:
            filter_string: Filter query string, defaults to searching all registered models.
            max_results: Maximum number of registered models desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_registered_models`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.RegisteredModel` objects
            that satisfy the search expressions. The pagination token for the next page can be
            obtained via the ``token`` attribute of the object.

        """
        if self._should_use_original():
            return super().search_registered_models(filter_string, max_results, order_by, page_token)

        from mlflow.store.entities.paged_list import PagedList

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(
            SearchRegisteredModels(
                filter=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(SearchRegisteredModels, req_body)
        registered_models = [
            RegisteredModel.from_proto(registered_model)
            for registered_model in response_proto.registered_models
        ]
        return PagedList(registered_models, response_proto.next_page_token)

    @override
    def get_latest_versions(self, name, stages=None, project_id: str | None = None):
        """
        Latest version models for each requested stage. If no ``stages`` argument is provided,
        returns the latest version for each stage.

        Args:
            name: Registered model name.
            stages: List of desired stages. If input list is None, return latest versions for
                each stage.

        Returns:
            List of :py:class:`mlflow.entities.model_registry.ModelVersion` objects.
        """
        if self._should_use_original():
            return super().get_latest_versions(name, stages)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(GetLatestVersions(name=name, stages=stages))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(GetLatestVersions, req_body, call_all_endpoints=True)
        return [
            ModelVersion.from_proto(model_version)
            for model_version in response_proto.model_versions
        ]

    @override
    def set_registered_model_tag(self, name, tag, project_id: str | None = None):
        """
        Set a tag for the registered model.

        Args:
            name: Registered model name.
            tag: :py:class:`mlflow.entities.model_registry.RegisteredModelTag` instance to log.

        Returns:
            None
        """
        if self._should_use_original():
            super().set_registered_model_tag(name, tag)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(SetRegisteredModelTag(name=name, key=tag.key, value=tag.value))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(SetRegisteredModelTag, req_body)

    @override
    def delete_registered_model_tag(self, name, key, project_id: str | None = None):
        """
        Delete a tag associated with the registered model.

        Args:
            name: Registered model name.
            key: Registered model tag key.

        Returns:
            None
        """
        if self._should_use_original():
            super().delete_registered_model_tag(name, key)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(DeleteRegisteredModelTag(name=name, key=key))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(DeleteRegisteredModelTag, req_body)

    @override
    def set_registered_model_alias(self, name, alias, version, project_id: str | None = None):
        """
        Set a registered model alias pointing to a model version.

        Args:
            name: Registered model name.
            alias: Name of the alias.
            version: Registered model version number.

        Returns:
            None
        """
        if self._should_use_original():
            super().set_registered_model_alias(name, alias, version)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(
            SetRegisteredModelAlias(name=name, alias=alias, version=str(version))
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(SetRegisteredModelAlias, req_body)

    @override
    def delete_registered_model_alias(self, name, alias, project_id: str | None = None):
        """
        Delete an alias associated with a registered model.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            None
        """
        if self._should_use_original():
            super().delete_registered_model_alias(name, alias)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(DeleteRegisteredModelAlias(name=name, alias=alias))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(DeleteRegisteredModelAlias, req_body)

    @override
    def create_model_version(self, name, source, run_id=None, tags=None, run_link=None, description=None, local_model_path=None, model_id: str | None = None, project_id: str | None = None):
        """
        Create a new model version from given source and run ID.

        Args:
            name: Registered model name.
            source: URI indicating the location of the model artifacts.
            run_id: Run ID from MLflow tracking server that generated the model.
            tags: A list of :py:class:`mlflow.entities.model_registry.ModelVersionTag`
                instances associated with this model version.
            run_link: Link to the run from an MLflow tracking server that generated this model.
            description: Description of the version.
            local_model_path: Unused.
            model_id: The ID of the model (from an Experiment) that is being promoted to a
                registered model version, if applicable.

        Returns:
            A single object of :py:class:`mlflow.entities.model_registry.ModelVersion`
            created in the backend.

        """
        if self._should_use_original():
            return super().create_model_version(name, source, run_id, tags, run_link, description, local_model_path, model_id)

        resolved_project_id = _resolve_project_id(project_id)
        proto_tags = [tag.to_proto() for tag in tags or []]
        req_body = message_to_json(
            CreateModelVersion(
                name=name,
                source=source,
                run_id=run_id,
                run_link=run_link,
                tags=proto_tags,
                description=description,
                model_id=model_id,
            )
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(CreateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    @override
    def transition_model_version_stage(self, name, version, stage, archive_existing_versions, project_id: str | None = None):
        """
        Update model version stage.

        Args:
            name: Registered model name.
            version: Registered model version.
            stage: New desired stage for this model version.
            archive_existing_versions: If this flag is set to ``True``, all existing model
                versions in the stage will be automatically moved to the "archived" stage. Only
                valid when ``stage`` is ``"staging"`` or ``"production"`` otherwise an error will
                be raised.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.

        """
        if self._should_use_original():
            return super().transition_model_version_stage(name, version, stage, archive_existing_versions)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(
            TransitionModelVersionStage(
                name=name,
                version=str(version),
                stage=stage,
                archive_existing_versions=archive_existing_versions,
            )
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(TransitionModelVersionStage, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    @override
    def update_model_version(self, name, version, description, project_id: str | None = None):
        """
        Update metadata associated with a model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.
            description: New model description.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        if self._should_use_original():
            return super().update_model_version(name, version, description)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(
            UpdateModelVersion(name=name, version=str(version), description=description)
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(UpdateModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    @override
    def delete_model_version(self, name, version, project_id: str | None = None):
        """
        Delete model version in backend.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            None
        """
        if self._should_use_original():
            super().delete_model_version(name, version)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(DeleteModelVersion(name=name, version=str(version)))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(DeleteModelVersion, req_body)

    @override
    def get_model_version(self, name, version, project_id: str | None = None):
        """
        Get the model version instance by name and version.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        if self._should_use_original():
            return super().get_model_version(name, version)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(GetModelVersion(name=name, version=str(version)))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(GetModelVersion, req_body)
        return ModelVersion.from_proto(response_proto.model_version)

    @override
    def get_model_version_download_uri(self, name, version, project_id: str | None = None):
        """
        Get the download location in Model Registry for this model version.
        NOTE: For first version of Model Registry, since the models are not copied over to another
              location, download URI points to input source path.

        Args:
            name: Registered model name.
            version: Registered model version.

        Returns:
            A single URI location that allows reads for downloading.
        """
        if self._should_use_original():
            return super().get_model_version_download_uri(name, version)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(GetModelVersionDownloadUri(name=name, version=str(version)))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(GetModelVersionDownloadUri, req_body)
        return response_proto.artifact_uri

    @override
    def search_model_versions(self, filter_string=None, max_results=None, order_by=None, page_token=None, project_id: str | None = None):
        """
        Search for model versions in backend that satisfy the filter criteria.

        Args:
            filter_string: A filter string expression. Currently supports a single filter
                condition either name of model like ``name = 'model_name'`` or
                ``run_id = '...'``.
            max_results: Maximum number of model versions desired.
            order_by: List of column names with ASC|DESC annotation, to be used for ordering
                matching search results.
            page_token: Token specifying the next page of results. It should be obtained from
                a ``search_model_versions`` call.

        Returns:
            A PagedList of :py:class:`mlflow.entities.model_registry.ModelVersion`
            objects that satisfy the search expressions. The pagination token for the next
            page can be obtained via the ``token`` attribute of the object.

        """
        if self._should_use_original():
            return super().search_model_versions(filter_string, max_results, order_by, page_token)

        from mlflow.store.entities.paged_list import PagedList

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(
            SearchModelVersions(
                filter=filter_string,
                max_results=max_results,
                order_by=order_by,
                page_token=page_token,
            )
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(SearchModelVersions, req_body)
        model_versions = [ModelVersion.from_proto(mvd) for mvd in response_proto.model_versions]
        return PagedList(model_versions, response_proto.next_page_token)

    @override
    def set_model_version_tag(self, name, version, tag, project_id: str | None = None):
        """
        Set a tag for the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            tag: :py:class:`mlflow.entities.model_registry.ModelVersionTag` instance to log.

        Returns:
            None
        """
        if self._should_use_original():
            super().set_model_version_tag(name, version, tag)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(
            SetModelVersionTag(name=name, version=str(version), key=tag.key, value=tag.value)
        )
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(SetModelVersionTag, req_body)

    @override
    def delete_model_version_tag(self, name, version, key, project_id: str | None = None):
        """
        Delete a tag associated with the model version.

        Args:
            name: Registered model name.
            version: Registered model version.
            key: Tag key.

        Returns:
            None
        """
        if self._should_use_original():
            super().delete_model_version_tag(name, version, key)
            return

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(DeleteModelVersionTag(name=name, version=str(version), key=key))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        self._call_endpoint(DeleteModelVersionTag, req_body)

    @override
    def get_model_version_by_alias(self, name, alias, project_id: str | None = None):
        """
        Get the model version instance by name and alias.

        Args:
            name: Registered model name.
            alias: Name of the alias.

        Returns:
            A single :py:class:`mlflow.entities.model_registry.ModelVersion` object.
        """
        if self._should_use_original():
            return super().get_model_version_by_alias(name, alias)

        resolved_project_id = _resolve_project_id(project_id)
        req_body = message_to_json(GetModelVersionByAlias(name=name, alias=alias))
        req_body = _inject_project_id_into_body(req_body, resolved_project_id)
        response_proto = self._call_endpoint(GetModelVersionByAlias, req_body)
        return ModelVersion.from_proto(response_proto.model_version)
