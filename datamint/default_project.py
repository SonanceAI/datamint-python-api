import threading
from typing import TYPE_CHECKING

from datamint import Api

if TYPE_CHECKING:
    from datamint.api.client import Api as ApiType
    from datamint.entities.project import Project

_lock = threading.Lock()
_default_project_hint: str | None = None


def select_project(project: 'Project | str', api: 'ApiType | None' = None) -> 'Project':
    """Set the default project for the current session.

    Args:
        project: The Project instance or project name/ID to set as default.
        api: The Api instance to resolve `project` against and to cache the
            resolved Project on directly.

    Returns:
        The resolved Project instance.
    """
    global _default_project_hint

    owns_api = api is None
    api = api or Api(check_connection=False)
    try:
        if isinstance(project, str):
            resolved = api.projects._get_by_name_or_id(project)
            if resolved is None:
                from datamint.exceptions import ItemNotFoundError
                raise ItemNotFoundError('Project', {'name_or_id': project})
        else:
            resolved = project

        with _lock:
            _default_project_hint = resolved.id

        if not owns_api:
            api.projects._default_project = resolved

        return resolved
    finally:
        if owns_api:
            api.close()


def get_default_project_hint() -> str | None:
    """Return the current default-project hint (a project name or ID), or None."""
    with _lock:
        return _default_project_hint


def clear_default_project() -> None:
    """Clear the current default-project hint."""
    global _default_project_hint
    with _lock:
        _default_project_hint = None
