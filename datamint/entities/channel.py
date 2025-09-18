from pydantic import ConfigDict, BaseModel
from datetime import datetime
from datamint.entities.base_entity import BaseEntity


class ChannelResourceData(BaseModel):
    """Represents resource data within a channel.
    
    Attributes:
        created_by: Email of the user who created the resource.
        customer_id: UUID of the customer.
        resource_id: UUID of the resource.
        resource_file_name: Original filename of the resource.
        resource_mimetype: MIME type of the resource.
    """
    model_config = ConfigDict(extra='allow')

    created_by: str
    customer_id: str
    resource_id: str
    resource_file_name: str
    resource_mimetype: str


class Channel(BaseEntity):
    """Represents a channel containing multiple resources.
    
    A channel is a collection of resources grouped together,
    typically for batch processing or organization purposes.
    
    Attributes:
        channel_name: Name identifier for the channel.
        resource_data: List of resources contained in this channel.
        deleted: Whether the channel has been marked as deleted.
        created_at: Timestamp when the channel was created.
        updated_at: Timestamp when the channel was last updated.
    """
    channel_name: str
    resource_data: list[ChannelResourceData]
    deleted: bool = False
    created_at: str | None = None
    updated_at: str | None = None

    def get_resource_ids(self) -> list[str]:
        """Get list of all resource IDs in this channel."""
        return [resource.resource_id for resource in self.resource_data] if self.resource_data else []