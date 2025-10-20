"""Project entity module for DataMint API."""

from datetime import datetime
import logging
from .base_entity import BaseEntity, MISSING_FIELD

logger = logging.getLogger(__name__)


class DatasetInfo(BaseEntity):
    """Pydantic Model representing a DataMint dataset.
    """

    id: str
    name: str
    created_at: str  # ISO timestamp string
    created_by: str
    description: str
    customer_id: str
    updated_at: str | None
    total_resource: int
    resource_ids: list[str]
