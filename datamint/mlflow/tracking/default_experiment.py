import os
import sys
from mlflow.tracking.default_experiment.abstract_context import (
    DefaultExperimentProvider,
)
from typing_extensions import override


class DatamintExperimentProvider(DefaultExperimentProvider):
    _experiment_id: str | None = None

    @override
    def in_context(self):  # type: ignore[override]
        return True

    @override
    def get_experiment_id(self):  # type: ignore[override]
        from mlflow.tracking.client import MlflowClient
        from datamint.mlflow.tracking.fluent import get_active_project_name

        if DatamintExperimentProvider._experiment_id is not None:
            return DatamintExperimentProvider._experiment_id

        # Prefer the active project (set via `datamint.mlflow.set_project()`) as the
        # experiment name. Fall back to the main source file's name.
        experiment_name = get_active_project_name() or os.path.basename(sys.argv[0])

        mlflowclient = MlflowClient()
        exp = mlflowclient.get_experiment_by_name(experiment_name)
        if exp is None:
            experiment_id = mlflowclient.create_experiment(experiment_name)
        else:
            experiment_id = exp.experiment_id
        DatamintExperimentProvider._experiment_id = experiment_id

        return experiment_id
