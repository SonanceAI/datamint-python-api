import logging
import os
import threading
from typing import TYPE_CHECKING

from datamint import Api
from datamint.exceptions import ItemNotFoundError
from datamint.mlflow.env_utils import ensure_mlflow_configured
from datamint.mlflow.env_vars import EnvVars

if TYPE_CHECKING:
    from datamint.entities.project import Project

_PROJECT_LOCK = threading.Lock()
_LOGGER = logging.getLogger(__name__)

_ACTIVE_PROJECT_ID: str | None = None
_ACTIVE_PROJECT_NAME: str | None = None


def get_active_project_id() -> str | None:
    """
    Get the active project ID from the environment variable or the global variable.
    """
    global _ACTIVE_PROJECT_ID, _ACTIVE_PROJECT_NAME

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
            _ACTIVE_PROJECT_NAME = project_name
            return _ACTIVE_PROJECT_ID

    return None


def get_active_project_name() -> str | None:
    """
    Get the active project's name, if one was set via `set_project()` or resolved
    from `DATAMINT_PROJECT_NAME`/`DATAMINT_PROJECT_ID`.
    """
    if _ACTIVE_PROJECT_NAME is None:
        get_active_project_id()
    return _ACTIVE_PROJECT_NAME


def _find_project_by_name(project_name: str):
    dt_client = Api(check_connection=False)
    project = dt_client.projects.get_by_name(project_name)
    if project is None:
        raise ItemNotFoundError('Project', {'name': project_name})
    return project


def _get_project_by_name_or_id(project_name_or_id: str) -> 'Project':
    dt_client = Api(check_connection=False)
    # If length == 36, likely an ID
    if len(project_name_or_id) == 36 and ' ' not in project_name_or_id:
        # Try to get by ID first
        try:
            project = dt_client.projects.get_by_id(project_name_or_id)
            if project is not None:
                return project
        except ValueError:
            pass  # Not a valid UUID, treat as name
    project = dt_client.projects.get_by_name(project_name_or_id)
    if project is None:
        raise ItemNotFoundError('Project', {'name_or_id': project_name_or_id})
    return project


def set_project(project: 'Project | str'):
    """
    Set the active project for the current session.

    The project's name also becomes the default MLflow experiment name for
    any run started without an explicit experiment (see `DatamintExperimentProvider`).

    Args:
        project: The Project instance or project name/ID to set as active.
    """
    global _ACTIVE_PROJECT_ID, _ACTIVE_PROJECT_NAME

    # Ensure MLflow is properly configured before proceeding
    ensure_mlflow_configured()

    with _PROJECT_LOCK:
        if isinstance(project, str):
            project = _get_project_by_name_or_id(project)

        _ACTIVE_PROJECT_ID = project.id
        _ACTIVE_PROJECT_NAME = project.name
        _reset_default_experiment_cache()

    # Set 'DATAMINT_PROJECT_ID' environment variable
    # so that subprocess can inherit it.
    os.environ[EnvVars.DATAMINT_PROJECT_ID.value] = project.id

    return project


def _reset_active_project():
    """Clear the active project, restoring the pre-``set_project()`` state. """
    global _ACTIVE_PROJECT_ID, _ACTIVE_PROJECT_NAME

    with _PROJECT_LOCK:
        _ACTIVE_PROJECT_ID = None
        _ACTIVE_PROJECT_NAME = None
        _reset_default_experiment_cache()

    os.environ.pop(EnvVars.DATAMINT_PROJECT_ID.value, None)


def _reset_default_experiment_cache():
    """Invalidate the cached default-experiment id so a project switch takes effect
    on the next run started without an explicit experiment."""
    from datamint.mlflow.tracking.default_experiment import DatamintExperimentProvider
    DatamintExperimentProvider._experiment_id = None
