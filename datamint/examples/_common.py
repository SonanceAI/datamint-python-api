import logging
from pathlib import Path

from datamint import Api
from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)


def get_or_create_project(project_name: str,
                          description: str,
                          api: Api) -> tuple[Project, bool]:
    """Return (project, already_existed).

    If a project with this name already exists, it is returned as-is (not modified).
    Otherwise a new, empty project is created for the caller to populate.
    """
    existing = api.projects.get_by_name(project_name)
    if existing is not None:
        return existing, True

    proj = api.projects.create(name=project_name, description=description, exists_ok=True)
    return proj, False


def print_skip_summary(dataset_name: str, proj: Project) -> None:
    print(dataset_name)
    print(f"  project '{proj.name}' already exists, skipping data population.")
    print(f'  {proj.url}')


def print_summary(dataset_name: str,
                  n_files: int,
                  n_annotated: int,
                  cache_path: Path,
                  proj: Project) -> None:
    pct = (n_annotated / n_files * 100) if n_files else 0.0
    print(dataset_name)
    print(f'  {n_files} files uploaded, {n_annotated} annotated ({pct:.0f}%)')
    print(f'  cached at {cache_path}')
    print(f'  project: {proj.name} ({proj.url})')
