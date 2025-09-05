import logging
from typing import Any, Optional, TypeVar, Generic, Type, Sequence, Generator
import httpx
from dataclasses import dataclass
from datamint.entities.base_entity import BaseEntity

logger = logging.getLogger(__name__)

# Generic type for entities
T = TypeVar('T', bound=BaseEntity)
_PAGE_LIMIT = 5000


@dataclass
class ApiConfig:
    """Configuration for API client."""
    base_url: str
    api_key: Optional[str] = None
    timeout: float = 30.0
    max_retries: int = 3


class BaseApi:
    """Base class for all API endpoint handlers."""

    def __init__(self, config: ApiConfig, client: Optional[httpx.Client] = None) -> None:
        """Initialize the base API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        self.config = config
        self.client = client or self._create_client()

    def _create_client(self) -> httpx.Client:
        """Create and configure HTTP client with authentication and timeouts."""
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["apikey"] = self.config.api_key

        return httpx.Client(
            base_url=self.config.base_url,
            headers=headers,
            timeout=self.config.timeout
        )

    def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request with error handling and retries.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for the request

        Returns:
            HTTP response object

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        url = endpoint.lstrip('/')  # Remove leading slash for httpx

        try:
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {method} {endpoint}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {method} {endpoint}: {e}")
            raise

    def _make_request_with_pagination(self,
                                      method: str,
                                      endpoint: str,
                                      return_field: str | None = None,
                                      **kwargs
                                      ) -> Generator[tuple[httpx.Response, dict | list | str], None, None]:
        offset = 0
        params = kwargs.get('params', {})
        while True:
            params['offset'] = offset
            params['limit'] = _PAGE_LIMIT

            response = self._make_request(method=method,
                                          endpoint=endpoint,
                                          **kwargs)
            items = self._convert_response(response.json(), return_field=return_field)
            yield response, items

            if len(items) < _PAGE_LIMIT:
                break

            offset += _PAGE_LIMIT

    def _convert_response(self,
                          data: dict | list,
                          return_field: str | None = None) -> list | dict | str:

        if isinstance(data, list):
            items = data
        else:
            if 'data' in data:
                items = data['data']
            elif 'items' in data:
                items = data['items']
            else:
                return data
            if return_field is not None:
                if 'totalCount' in data and len(items) == 1 and return_field in items[0]:
                    items = items[0][return_field]
        return items


class EntityBaseApi(BaseApi, Generic[T]):
    """Base API handler for entity-related endpoints with CRUD operations.

    This class provides a template for API handlers that work with specific
    entity types, offering common CRUD operations with proper typing.

    Type Parameters:
        T: The entity type this API handler manages (must extend BaseEntity)
    """

    def __init__(self, config: ApiConfig,
                 entity_class: Type[T],
                 endpoint_base: str,
                 client: Optional[httpx.Client] = None) -> None:
        """Initialize the entity API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            entity_class: The entity class this handler manages
            endpoint_base: Base endpoint path (e.g., 'projects', 'annotations')
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        super().__init__(config, client)
        self.entity_class = entity_class
        self.endpoint_base = endpoint_base.strip('/')

    def get_list(self, **kwargs) -> Sequence[T]:
        """Get entities with optional filtering.

        Returns:
            List of entity instances

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        params = dict(kwargs)

        # Remove None values from the payload.
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        # response = self._make_request('GET', f'/{self.endpoint_base}',
        #                               params=params)
        # items = self._convert_response(response.json())
        items_gen = self._make_request_with_pagination('GET', f'/{self.endpoint_base}',
                                                   return_field=self.endpoint_base,
                                                   params=params)
        

        return [self.entity_class(**item) for resp,items in items_gen for item in items]

    def get_all(self) -> Sequence[T]:
        """Get all entities with optional pagination and filtering.

        Returns:
            List of entity instances

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        return self.get_list()

    def get_by_id(self, entity_id: str) -> T:
        """Get a specific entity by its ID.

        Args:
            entity_id: Unique identifier for the entity

        Returns:
            Entity instance

        Raises:
            httpx.HTTPStatusError: If the entity is not found or request fails
        """
        response = self._make_request('GET', f'/{self.endpoint_base}/{entity_id}')
        return self.entity_class(**response.json())

    def create(self, entity_data: dict[str, Any]) -> T:
        """Create a new entity.

        Args:
            entity_data: Dictionary containing entity data for creation

        Returns:
            Created entity instance

        Raises:
            httpx.HTTPStatusError: If creation fails
        """
        response = self._make_request('POST', f'/{self.endpoint_base}', json=entity_data)
        return self.entity_class(**response.json())

    def update(self, entity_id: str, entity_data: dict[str, Any]) -> T:
        """Update an existing entity.

        Args:
            entity_id: Unique identifier for the entity
            entity_data: Dictionary containing updated entity data

        Returns:
            Updated entity instance

        Raises:
            httpx.HTTPStatusError: If update fails or entity not found
        """
        response = self._make_request('PUT', f'/{self.endpoint_base}/{entity_id}', json=entity_data)
        return self.entity_class(**response.json())

    def delete(self, entity_id: str) -> None:
        """Delete an entity by its ID.

        Args:
            entity_id: Unique identifier for the entity to delete

        Raises:
            httpx.HTTPStatusError: If deletion fails or entity not found
        """
        self._make_request('DELETE', f'/{self.endpoint_base}/{entity_id}')

    # def bulk_create(self, entities_data: list[dict[str, Any]]) -> list[T]:
    #     """Create multiple entities in a single request.

    #     Args:
    #         entities_data: List of dictionaries containing entity data

    #     Returns:
    #         List of created entity instances

    #     Raises:
    #         httpx.HTTPStatusError: If bulk creation fails
    #     """
    #     payload = {'items': entities_data}  # Common bulk API format
    #     response = self._make_request('POST', f'/{self.endpoint_base}/bulk', json=payload)
    #     data = response.json()

    #     # Handle response format - may be direct list or wrapped
    #     items = data if isinstance(data, list) else data.get('items', [])
    #     return [self.entity_class(**item) for item in items]

    # def count(self, **params: Any) -> int:
    #     """Get the total count of entities matching the given filters.

    #     Args:
    #         **params: Query parameters for filtering

    #     Returns:
    #         Total count of matching entities

    #     Raises:
    #         httpx.HTTPStatusError: If the request fails
    #     """
    #     response = self._make_request('GET', f'/{self.endpoint_base}/count', params=params)
    #     data = response.json()
    #     return data.get('count', 0) if isinstance(data, dict) else data
