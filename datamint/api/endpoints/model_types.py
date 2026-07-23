"""Thin wrapper objects over MLflow's model registry entities."""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow.models
from mlflow.entities.model_registry import ModelVersion as MlflowModelVersion
from mlflow.entities.model_registry import RegisteredModel as MlflowRegisteredModel

from datamint.entities.annotations.annotation_spec import AnnotationSpec
from datamint.mlflow.flavors.datamint_flavor import FLAVOR_NAME
from datamint.mlflow.models.tags import DATAMINT_LOGGED_MODEL_ID_TAG

if TYPE_CHECKING:
    from .models_api import ModelsApi


@dataclass
class ModelVersion:
    """A single version of a registered model."""

    _raw: MlflowModelVersion
    _api: 'ModelsApi'

    @property
    def name(self) -> str:
        return self._raw.name

    @property
    def version(self) -> str:
        return self._raw.version

    @property
    def run_id(self) -> str | None:
        return self._raw.run_id

    @property
    def creation_timestamp(self) -> int:
        return self._raw.creation_timestamp

    @property
    def source(self) -> str | None:
        return self._raw.source

    @property
    def current_stage(self) -> str | None:
        return self._raw.current_stage

    @property
    def aliases(self) -> list[str]:
        return self._raw.aliases

    @property
    def tags(self) -> dict[str, str]:
        return self._raw.tags

    def _flavor_data(self) -> dict:
        if self.source is None:
            return {}
        model_info = mlflow.models.get_model_info(self.source)
        return model_info.flavors.get(FLAVOR_NAME, {})

    def get_supported_modes(self) -> list[str]:
        """Prediction modes this model version supports (from the ``datamint`` flavor)."""
        return self._flavor_data().get('supported_modes', [])

    def get_task_type(self) -> str | None:
        """Task type this model version was trained for, or ``None`` if not recorded."""
        return self._flavor_data().get('task_type')

    def get_annotation_specs(self) -> list[AnnotationSpec] | None:
        """Annotation specs this model version produces, or ``None`` if not recorded."""
        raw_specs = self._flavor_data().get('annotation_specs')
        if not raw_specs:
            return None
        return [AnnotationSpec.create(**s) for s in raw_specs]

    def get_metrics(self) -> dict[str, float]:
        """Training/test metrics logged for this version.

        Returns ``{}`` when this version has no ``DATAMINT_LOGGED_MODEL_ID_TAG``
        (e.g. an externally-registered model with no Datamint-trained run behind it).
        """
        logged_model_id = self.tags.get(DATAMINT_LOGGED_MODEL_ID_TAG)
        if logged_model_id is None:
            return {}
        logged_model = mlflow.get_logged_model(logged_model_id)
        return {m.key: m.value for m in logged_model.metrics}

    def is_deployed(self) -> bool:
        return self._api._deploy_api.image_exists(self.name)


@dataclass
class Model:
    """A registered model: a named family of :class:`ModelVersion`."""

    _raw: MlflowRegisteredModel
    _api: 'ModelsApi'

    @property
    def name(self) -> str:
        return self._raw.name

    @property
    def description(self) -> str | None:
        return self._raw.description

    @property
    def creation_timestamp(self) -> int:
        return self._raw.creation_timestamp

    @property
    def last_updated_timestamp(self) -> int:
        return self._raw.last_updated_timestamp

    @property
    def tags(self) -> dict[str, str]:
        return self._raw.tags

    def get_versions(self) -> list[ModelVersion]:
        raw_versions = self._api._mlflow_client.search_model_versions(f"name='{self.name}'")
        return [ModelVersion(_raw=v, _api=self._api) for v in raw_versions]

    def get_latest_version(self, alias: str | None = None) -> ModelVersion | None:
        """Most recently created version, or the version at *alias* if given."""
        if alias is not None:
            raw = self._api._mlflow_client.get_model_version_by_alias(self.name, alias)
            return ModelVersion(_raw=raw, _api=self._api) if raw else None
        versions = self.get_versions()
        if not versions:
            return None
        return max(versions, key=lambda v: int(v.version))

    def get_supported_modes(self, version: ModelVersion | None = None) -> list[str]:
        version = version or self.get_latest_version()
        return version.get_supported_modes() if version else []

    def get_metrics(self, version: ModelVersion | None = None) -> dict[str, float]:
        version = version or self.get_latest_version()
        return version.get_metrics() if version else {}

    def is_deployed(self) -> bool:
        return self._api._deploy_api.image_exists(self.name)
