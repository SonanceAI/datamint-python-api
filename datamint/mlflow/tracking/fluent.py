from typing import Optional
from mlflow.exceptions import MlflowException
import threading
import logging
from datamint import APIHandler
from datamintapi.apihandler.base_api_handler import DatamintException
import os
from datamint.mlflow.env_vars import EnvVars

_PROJECT_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_ACTIVE_PROJECT_ID: Optional[str] = None


def get_active_project_id() -> str | None:
    """
    Get the active project ID from the environment variable or the global variable.
    """
    global _ACTIVE_PROJECT_ID

    if _ACTIVE_PROJECT_ID is not None:
        return _ACTIVE_PROJECT_ID
    # Check if the environment variable is set
    project_id = os.getenv(EnvVars.DATAMINT_PROJECT_ID.value)
    if project_id is not None:
        _ACTIVE_PROJECT_ID = project_id
        return project_id
    project_name = os.getenv(EnvVars.DATAMINT_PROJECT_NAME.value)
    if project_name is not None:
        project = _find_project_by_name(project_name)
        if project is not None:
            _ACTIVE_PROJECT_ID = project['id']
            return _ACTIVE_PROJECT_ID

    return None


def _find_project_by_name(project_name: str) -> Optional[dict]:
    dt_client = APIHandler(check_connection=False)
    project = dt_client.get_project_by_name(project_name)
    if project is None:
        raise DatamintException(f"Project with name '{project_name}' does not exist.")
    if 'error' in project:
        raise DatamintException(f'Error getting project: {project["error"]}')
    return project


def set_project(project_name: Optional[str] = None, project_id: Optional[str] = None) -> dict:
    global _ACTIVE_PROJECT_ID
    if project_name is None and project_id is None:
        raise MlflowException("You must specify either a project name or a project id")

    if project_name is not None and project_id is not None:
        raise MlflowException("You cannot specify both a project name and a project id")

    with _PROJECT_LOCK:
        dt_client = APIHandler(check_connection=False)
        if project_id is None:
            project = dt_client.get_project_by_name(project_name)
            if project is None:
                raise DatamintException(f"Project with name '{project_name}' does not exist.")
            if 'error' in project:
                raise DatamintException(f'Error getting project "{project_name}" by name: {project["error"]}')
            project_id = project['id']
        else:
            project = dt_client.get_project_by_id(project_id)
            if project is None:
                raise DatamintException(f"Project with id '{project_id}' does not exist.")

        _ACTIVE_PROJECT_ID = project_id

    # Set 'DATAMINT_PROJECT_ID' environment variable
    # so that subprocess can inherit it.
    os.environ[EnvVars.DATAMINT_PROJECT_ID.value] = project_id

    return project
