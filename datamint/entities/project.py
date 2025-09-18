"""Project entity module for DataMint API."""

from datetime import datetime
import logging
from .base_entity import BaseEntity, MISSING_FIELD

logger = logging.getLogger(__name__)


class Project(BaseEntity):
    """Pydantic Model representing a DataMint project.

    This class models a project entity from the DataMint API, containing
    information about the project, its dataset, worklist, AI model, and
    annotation statistics.

    Attributes:
        id: Unique identifier for the project
        name: Human-readable name of the project
        description: Optional description of the project
        created_at: ISO timestamp when the project was created
        created_by: Email of the user who created the project
        dataset_id: ID of the associated dataset
        worklist_id: ID of the associated worklist
        ai_model_id: Optional ID of the associated AI model
        viewable_ai_segs: Optional configuration for viewable AI segments
        editable_ai_segs: Optional configuration for editable AI segments
        archived: Whether the project is archived
        resource_count: Total number of resources in the project
        annotated_resource_count: Number of resources that have been annotated
        most_recent_experiment: Optional information about the most recent experiment
        closed_resources_count: Number of resources marked as closed/completed
        resources_to_annotate_count: Number of resources still needing annotation
        annotators: List of annotators assigned to this project
    """
    id: str
    name: str
    created_at: str  # ISO timestamp string
    created_by: str
    dataset_id: str
    worklist_id: str
    archived: bool
    resource_count: int
    annotated_resource_count: int
    description: str | None
    ai_model_id: str | None
    viewable_ai_segs: list | None
    editable_ai_segs: list | None
    closed_resources_count: int = MISSING_FIELD
    resources_to_annotate_count: int = MISSING_FIELD
    most_recent_experiment: str | None = MISSING_FIELD  # ISO timestamp string
    annotators: list[dict] = MISSING_FIELD
    customer_id: str | None = MISSING_FIELD
    archived_on: str | None = MISSING_FIELD
    archived_by: str | None = MISSING_FIELD
    is_active_learning: bool = MISSING_FIELD
    two_up_display: bool = MISSING_FIELD
    require_review: bool = MISSING_FIELD

    @property
    def url(self) -> str:
        """Get the URL to access this project in the DataMint web application."""
        base_url = "https://app.datamint.io/projects/edit"
        return f"{base_url}/{self.id}"
