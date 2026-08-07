import os
import sys
import os
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
        
        if DatamintExperimentProvider._experiment_id is not None:
            return DatamintExperimentProvider._experiment_id
        # Get the filename of the main source file
        source_code_filename = os.path.basename(sys.argv[0])
        mlflowclient = MlflowClient()
        exp = mlflowclient.get_experiment_by_name(source_code_filename)
        if exp is None:
            experiment_id = mlflowclient.create_experiment(source_code_filename)
        else:
            experiment_id = exp.experiment_id
        DatamintExperimentProvider._experiment_id = experiment_id

        return experiment_id
