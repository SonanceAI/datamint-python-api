from typing import Optional, TYPE_CHECKING
import threading
import logging
from datamint import Api
from datamint.exceptions import DatamintException
import os
from datamint.mlflow.env_vars import EnvVars
from datamint.mlflow.env_utils import ensure_mlflow_configured

if TYPE_CHECKING:
    from datamint.entities.project import Project

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


def _find_project_by_name(project_name: str):
    dt_client = Api(check_connection=False)
    project = dt_client.projects.get_by_name(project_name)
    if project is None:
        raise DatamintException(f"Project with name '{project_name}' does not exist.")
    return project


def _get_project_by_name_or_id(project_name_or_id: str) -> 'Project':
    dt_client = Api(check_connection=False)
    # If length >= 32, likely an ID
    if len(project_name_or_id) >= 32 and ' ' not in project_name_or_id:
        # Try to get by ID first
        project = dt_client.projects.get_by_id(project_name_or_id)
        if project is not None:
            return project
    project = dt_client.projects.get_by_name(project_name_or_id)
    if project is None:
        raise DatamintException(f"Project '{project_name_or_id}' does not exist.")
    return project


def set_project(project: 'Project | str'):
    """
    Set the active project for the current session.
    
    Args:
        project: The Project instance or project name/ID to set as active.
    """
    global _ACTIVE_PROJECT_ID

    # Ensure MLflow is properly configured before proceeding
    ensure_mlflow_configured()

    with _PROJECT_LOCK:
        if isinstance(project, str):
            project_id = None
            project = _get_project_by_name_or_id(project)
            project_id = project.id
        else:
            # It's a Project entity
            project_id = project.id

        _ACTIVE_PROJECT_ID = project_id

    # Set 'DATAMINT_PROJECT_ID' environment variable
    # so that subprocess can inherit it.
    os.environ[EnvVars.DATAMINT_PROJECT_ID.value] = project_id

    return project
