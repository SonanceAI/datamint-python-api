from functools import partial

import requests
from mlflow.exceptions import MlflowException
from mlflow.store.tracking.rest_store import RestStore
from mlflow.utils.proto_json_utils import message_to_json
from typing_extensions import override
from datamint.mlflow.store_utils import resolve_project_id, _inject_project_id_into_body
from datamint.mlflow.tracking.offline_buffer import OfflineLogBuffer
import logging

_LOGGER = logging.getLogger(__name__)
_TRANSIENT_NETWORK_CAUSES = (requests.exceptions.ConnectionError, requests.exceptions.Timeout)


def _is_transient_network_failure(exc: MlflowException) -> bool:
    cause = exc.__cause__ or exc.__context__
    return isinstance(cause, _TRANSIENT_NETWORK_CAUSES)


class DatamintStore(RestStore):
    """
    DatamintStore is a subclass of RestStore that provides a tracking store
    implementation for Datamint.
    """

    def __init__(self, store_uri: str, artifact_uri=None, force_valid=True):
        # Ensure MLflow environment is configured when store is initialized
        from mlflow.utils.credentials import get_default_host_creds

        from datamint.mlflow.env_utils import setup_mlflow_environment
        setup_mlflow_environment()

        if store_uri.startswith('datamint://') or 'datamint.io' in store_uri or force_valid:
            self.invalid = False
        else:
            self.invalid = True

        store_uri = store_uri.split('datamint://', maxsplit=1)[-1]
        get_host_creds = partial(get_default_host_creds, store_uri)
        super().__init__(get_host_creds=get_host_creds)

        self._warned_offline_runs: set[str] = set()

    def _buffer_after_network_failure(self, run_id: str, kind: str, data: dict) -> None:
        if run_id not in self._warned_offline_runs:
            self._warned_offline_runs.add(run_id)
            _LOGGER.warning(
                "Lost connection to the Datamint MLflow backend for run '%s'. "
                "Logging locally and will resume sending once the connection is back.",
                run_id,
            )
        OfflineLogBuffer(run_id).append(kind, data)

    @override
    def log_metric(self, run_id, metric):
        try:
            return super().log_metric(run_id, metric)
        except MlflowException as e:
            if not _is_transient_network_failure(e):
                raise
            self._buffer_after_network_failure(run_id, 'metric', {
                'key': metric.key,
                'value': metric.value,
                'timestamp': metric.timestamp,
                'step': metric.step,
            })

    @override
    def log_param(self, run_id, param):
        try:
            return super().log_param(run_id, param)
        except MlflowException as e:
            if not _is_transient_network_failure(e):
                raise
            self._buffer_after_network_failure(run_id, 'param', {
                'key': param.key,
                'value': param.value,
            })

    @override
    def log_batch(self, run_id, metrics=(), params=(), tags=()):
        try:
            return super().log_batch(run_id, metrics=metrics, params=params, tags=tags)
        except MlflowException as e:
            if not _is_transient_network_failure(e):
                raise
            for metric in metrics:
                self._buffer_after_network_failure(run_id, 'metric', {
                    'key': metric.key,
                    'value': metric.value,
                    'timestamp': metric.timestamp,
                    'step': metric.step,
                })
            for param in params:
                self._buffer_after_network_failure(run_id, 'param', {
                    'key': param.key,
                    'value': param.value,
                })
            for tag in tags:
                self._buffer_after_network_failure(run_id, 'tag', {
                    'key': tag.key,
                    'value': tag.value,
                })

    def sync_offline_logs(self, run_id: str) -> int:
        """
        Send everything buffered locally for `run_id` to the remote Datamint MLflow
        backend, then clear the buffer. Returns the number of entries synced.

        If the connection is still down, the underlying network exception propagates
        and the buffer is left untouched, so a later retry can pick up where this
        left off.
        """
        from mlflow.entities import Metric, Param, RunTag

        buffer = OfflineLogBuffer(run_id)
        entries = buffer.read_all()
        if not entries:
            return 0

        metrics, params, tags = [], [], []
        for entry in entries:
            kind, data = entry['kind'], entry['data']
            if kind == 'metric':
                metrics.append(Metric(key=data['key'], value=data['value'],
                                       timestamp=data['timestamp'], step=data['step']))
            elif kind == 'param':
                params.append(Param(key=data['key'], value=data['value']))
            elif kind == 'tag':
                tags.append(RunTag(key=data['key'], value=data['value']))
            else:
                _LOGGER.warning("Unknown buffered entry kind '%s' for run '%s', skipping.",
                                 kind, run_id)

        # Bypass our own buffering override: if this raises, the connection is
        # still down and the buffer must stay intact for a later retry.
        RestStore.log_batch(self, run_id, metrics=metrics, params=params, tags=tags)

        buffer.clear()
        self._warned_offline_runs.discard(run_id)
        _LOGGER.info("Synced %d buffered log entr%s for run '%s'.",
                     len(entries), 'y' if len(entries) == 1 else 'ies', run_id)
        return len(entries)

    def create_experiment(self, name, artifact_location=None, tags=None, project_id: str | None = None) -> str:
        from mlflow.protos.service_pb2 import CreateExperiment

        if self.invalid:
            return super().create_experiment(name, artifact_location, tags)

        resolved_project_id = resolve_project_id(project_id)
        tag_protos = [tag.to_proto() for tag in tags] if tags else []
        req_body = message_to_json(
            CreateExperiment(name=name, artifact_location=artifact_location, tags=tag_protos)
        )

        req_body = _inject_project_id_into_body(req_body, resolved_project_id)

        response_proto = self._call_endpoint(CreateExperiment, req_body)
        return response_proto.experiment_id

    @override
    def get_experiment_by_name(self, experiment_name, project_id: str | None = None):
        from mlflow.entities import Experiment
        from mlflow.protos import databricks_pb2
        from mlflow.protos.service_pb2 import GetExperimentByName

        if self.invalid:
            return super().get_experiment_by_name(experiment_name)

        resolved_project_id = resolve_project_id(project_id)
        try:
            req_body = message_to_json(GetExperimentByName(experiment_name=experiment_name))
            req_body = _inject_project_id_into_body(req_body, resolved_project_id)

            response_proto = self._call_endpoint(GetExperimentByName, req_body)
            return Experiment.from_proto(response_proto.experiment)
        except MlflowException as e:
            if e.error_code == databricks_pb2.ErrorCode.Name(
                databricks_pb2.RESOURCE_DOES_NOT_EXIST
            ):
                return None
            else:
                raise
