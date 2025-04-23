from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.credentials import get_default_host_creds
from functools import partial
from .fluent import get_active_project_id
from mlflow.utils.proto_json_utils import message_to_json
from mlflow.protos.service_pb2 import CreateExperiment
import json


class DatamintStore(RestStore):
    """
    DatamintStore is a subclass of RestStore that provides a tracking store
    implementation for Datamint.
    """

    def __init__(self, store_uri: str, artifact_uri=None, force_valid=False):
        if store_uri.startswith('datamint://') or 'datamint.io' in store_uri or force_valid:
            self.invalid = False
        else:
            self.invalid = True

        store_uri = store_uri.split('datamint://', maxsplit=1)[-1]
        get_host_creds = partial(get_default_host_creds, store_uri)
        super().__init__(get_host_creds=get_host_creds)

    def create_experiment(self, name, artifact_location=None, tags=None, project_id: str = None) -> str:
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
