from mlflow.store.tracking.rest_store import RestStore
from mlflow.exceptions import MlflowException
from mlflow.utils.proto_json_utils import message_to_json
from functools import partial
import json
from typing_extensions import override


class DatamintStore(RestStore):
    """
    DatamintStore is a subclass of RestStore that provides a tracking store
    implementation for Datamint.
    """

    def __init__(self, store_uri: str, artifact_uri=None, force_valid=True):
        # Ensure MLflow environment is configured when store is initialized
        from datamint.mlflow.env_utils import setup_mlflow_environment
        from mlflow.utils.credentials import get_default_host_creds
        setup_mlflow_environment()

        if store_uri.startswith('datamint://') or 'datamint.io' in store_uri or force_valid:
            self.invalid = False
        else:
            self.invalid = True

        store_uri = store_uri.split('datamint://', maxsplit=1)[-1]
        get_host_creds = partial(get_default_host_creds, store_uri)
        super().__init__(get_host_creds=get_host_creds)

    def create_experiment(self, name, artifact_location=None, tags=None, project_id: str | None = None) -> str:
        from mlflow.protos.service_pb2 import CreateExperiment
        from datamint.mlflow.tracking.fluent import get_active_project_id

        if self.invalid:
            return super().create_experiment(name, artifact_location, tags)
        if project_id is None:
            project_id = get_active_project_id()
        tag_protos = [tag.to_proto() for tag in tags] if tags else []
        req_body = message_to_json(
            CreateExperiment(name=name, artifact_location=artifact_location, tags=tag_protos)
        )

        req_body = json.loads(req_body)
        req_body["project_id"] = project_id  # FIXME: this should be in the proto
        req_body = json.dumps(req_body)

        response_proto = self._call_endpoint(CreateExperiment, req_body)
        return response_proto.experiment_id

    @override
    def get_experiment_by_name(self, experiment_name, project_id: str | None = None):
        from datamint.mlflow.tracking.fluent import get_active_project_id
        from mlflow.protos.service_pb2 import GetExperimentByName
        from mlflow.entities import Experiment
        from mlflow.protos import databricks_pb2

        if self.invalid:
            return super().get_experiment_by_name(experiment_name)
        if project_id is None:
            project_id = get_active_project_id()
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            if project_id:
                body = json.loads(req_body)
                body["project_id"] = project_id
                req_body = json.dumps(body)

            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.RESOURCE_DOES_NOT_EXIST
            ):
                return None
            else:
                raise
