from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.exceptions import MlflowException
import json
import logging

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
    """Inject the project_id into a protobuf JSON request body."""
    body = json.loads(req_body)
    body["project_id"] = project_id
    return json.dumps(body)
