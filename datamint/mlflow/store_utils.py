import json
import logging

from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

_LOGGER = logging.getLogger(__name__)


def resolve_project_id(project_id: str | None) -> str:
    """
    Resolve the project ID from the provided value or active context.

    Raises MlflowException if no active project is found.
    """
    from datamint.mlflow.tracking.fluent import get_active_project_id

    if project_id is None:
        project_id = get_active_project_id()
        if project_id is None:
            msg = ("No project_id provided and no active project found. "
                   "Please set the active project using `datamint.mlflow.set_project()` and "
                   "ensure it is called before `mlflow.set_experiment()` and `mlflow.start_run()`.")
            _LOGGER.error(msg)
            # print all exception stack trace
            raise MlflowException(
                message=msg,
                error_code=INVALID_PARAMETER_VALUE,
            )
    return project_id


def _inject_project_id_into_body(req_body: str, project_id: str) -> str:
    """Inject the project_id into a protobuf JSON request body, overwriting any existing value."""
    body = json.loads(req_body)
    body["project_id"] = project_id
    return json.dumps(body)


def add_active_project_id_if_absent(req_body: str) -> str:
    """Scope a serialized protobuf request body to the active Datamint project.

    Project scoping is expressed as an extra field on the request body. MLflow's
    protobuf messages do not declare this field, so it has to be spliced into the
    JSON produced by ``message_to_json``.

    A ``project_id`` already present in the body wins, so callers that resolve the
    project themselves (e.g. ``create_experiment(project_id=...)``) are not
    overridden. The active project is only looked up when the body is unscoped.

    Args:
        req_body: Serialized protobuf request body.

    Returns:
        The serialized request body, with ``project_id`` set when absent.

    Raises:
        MlflowException: If the body is unscoped and no project is active.
    """
    body = json.loads(req_body)
    if body.get("project_id") is not None:
        return req_body
    body["project_id"] = resolve_project_id(None)
    return json.dumps(body)
