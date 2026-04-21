from datamint.entities.base_entity import BaseEntityModel


class ProjectResourceSplit(BaseEntityModel):
    """Read-only DTO representing a resource-to-split assignment within a project."""

    id: str
    split_name: str
    project_id: str
    resource_id: str
    created_at: str
    created_by: str
    deleted_at: str | None = None
    deleted_by: str | None = None