"""
Channels API endpoint for managing channel resources.

This module provides functionality to interact with channels,
which are collections of resources grouped together for
batch processing or organization purposes.
"""

import logging
import httpx
from ..entity_base_api import EntityBaseApi
from datamint.entities.channel import Channel

logger = logging.getLogger(__name__)


class ChannelsApi(EntityBaseApi[Channel]):
    """API client for channel-related operations.
    """

    def __init__(self, config, client: httpx.Client | None = None) -> None:
        """Initialize the Channels API client.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, Channel, 'resources/channels', client)
