import logging
from typing import Any, TypeVar, Generic, Type, Sequence, Generator, Literal
import httpx
from dataclasses import dataclass
from datamint.entities.base_entity import BaseEntity
from datamint.exceptions import DatamintException, ResourceNotFoundError
import aiohttp
import json

logger = logging.getLogger(__name__)

# Generic type for entities
T = TypeVar('T', bound=BaseEntity)
_PAGE_LIMIT = 5000


@dataclass
class ApiConfig:
    """Configuration for API client.

    Attributes:
        server_url: Base URL for the API.
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for requests.
    """
    server_url: str
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3


class BaseApi:
    """Base class for all API endpoint handlers."""

    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        """Initialize the base API handler.

        Args:
            config: API configuration containing base URL, API key, etc.
            client: Optional HTTP client instance. If None, a new one will be created.
        """
        self.config = config
        self.client = client or self._create_client()

    def _create_client(self) -> httpx.Client:
        """Create and configure HTTP client with authentication and timeouts."""
        headers = None
        if self.config.api_key:
            headers = {"apikey": self.config.api_key}

        return httpx.Client(
            base_url=self.config.server_url,
            headers=headers,
            timeout=self.config.timeout
        )

    def _stream_request(self, method: str, endpoint: str, **kwargs):
        """Make streaming HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments for the request

        Returns:
            HTTP response object configured for streaming

        Raises:
            httpx.HTTPStatusError: If the request fails

        Example:
            with api._stream_request('GET', '/large-file') as response:
                for chunk in response.iter_bytes():
                    process_chunk(chunk)
        """
        url = endpoint.lstrip('/')  # Remove leading slash for httpx

        try:
            return self.client.stream(method, url, **kwargs)
        except httpx.RequestError as e:
            logger.error(f"Request error for streaming {method} {endpoint}: {e}")
            raise

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
            curl_command = self._generate_curl_command({"method": method,
                                                        "url": url,
                                                        "headers": self.client.headers,
                                                        **kwargs})
            logger.debug(f'Equivalent curl command: "{curl_command}"')
            response = self.client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error {e.response.status_code} for {method} {endpoint}: {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error for {method} {endpoint}: {e}")
            raise

    def _generate_curl_command(self, request_args: dict) -> str:
        """
        Generate a curl command for debugging purposes.

        Args:
            request_args (dict): Request arguments dictionary containing method, url, headers, etc.

        Returns:
            str: Equivalent curl command
        """
        method = request_args.get('method', 'GET').upper()
        url = request_args['url']
        headers = request_args.get('headers', {})
        data = request_args.get('json') or request_args.get('data')
        params = request_args.get('params')

        curl_command = ['curl']

        # Add method if not GET
        if method != 'GET':
            curl_command.extend(['-X', method])

        # Add headers
        for key, value in headers.items():
            if key.lower() == 'apikey':
                value = '<YOUR-API-KEY>'  # Mask API key for security
            curl_command.extend(['-H', f"'{key}: {value}'"])

        # Add query parameters
        if params:
            param_str = '&'.join([f"{k}={v}" for k, v in params.items()])
            url = f"{url}?{param_str}"
        # Add URL
        curl_command.append(f"'{url}'")

        # Add data
        if data:
            if isinstance(data, aiohttp.FormData):  # Check if it's aiohttp.FormData
                # Handle FormData by extracting fields
                form_parts = []
                for options, headers, value in data._fields:
                    # get the name from options
                    name = options.get('name', 'file')
                    if hasattr(value, 'read'):  # File-like object
                        filename = getattr(value, 'name', 'file')
                        form_parts.extend(['-F', f"'{name}=@{filename}'"])
                    else:
                        form_parts.extend(['-F', f"'{name}={value}'"])
                curl_command.extend(form_parts)
            elif isinstance(data, dict):
                curl_command.extend(['-d', f"'{json.dumps(data)}'"])
            else:
                curl_command.extend(['-d', f"'{data}'"])

        return ' '.join(curl_command)

    @staticmethod
    def get_status_code(e) -> int:
        if not hasattr(e, 'response') or e.response is None:
            return -1
        return e.response.status_code

    def _check_errors_response(self,
                               response,
                               url: str):
        try:
            if hasattr(response, 'raise_for_status'):
                response.raise_for_status()
        except Exception as e:
            status_code = BaseApi.get_status_code(e)
            if status_code >= 500 and status_code < 600:
                logger.error(f"Error in request to {url}: {e}")
            if status_code >= 400 and status_code < 500:
                try:
                    logger.info(f"Error response: {response.text}")
                    error_data = response.json()
                except Exception as e2:
                    logger.info(f"Error parsing the response. {e2}")
                else:
                    if isinstance(error_data['message'], str) and ' not found' in error_data['message'].lower():
                        # Will be caught by the caller and properly initialized:
                        raise ResourceNotFoundError('unknown', {})

            raise

    async def _make_request_async(self,
                                  method: str,
                                  endpoint: str,
                                  session: aiohttp.ClientSession | None = None,
                                  data_to_get: Literal['json', 'text', 'content'] = 'json',
                                  **kwargs):
        """Make asynchronous HTTP request with error handling.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            session: Optional aiohttp session. If None, a new one will be created.
            **kwargs: Additional arguments for the request

        Returns:
            HTTP response object

        Raises:
            aiohttp.ClientError: If the request fails
        """
        url = f"{self.config.server_url.rstrip('/')}/{endpoint.lstrip('/')}"

        # Prepare headers
        headers = kwargs.pop('headers', {})
        if self.config.api_key:
            headers['apikey'] = self.config.api_key

        # Set timeout
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        async def make_request(client_session: aiohttp.ClientSession) -> aiohttp.ClientResponse | Any:
            try:
                # logger.debug(f"Making async {method} request to {url} with headers {headers} and kwargs:\n {kwargs}")
                try:
                    logger.debug(f"Running request to {url}")
                    logger.debug(f'Equivalent curl command: "{self._generate_curl_command({"method": method,
                                                                                           "url": url,
                                                                                           "headers": headers,
                                                                                           **kwargs})}"'
                                 )
                except Exception as e:
                    logger.debug(f"Error generating curl command: {e}")
                async with client_session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs
                ) as response:
                    # Check for HTTP errors
                    # if response.status >= 400:
                    #     error_text = await response.text()
                    #     logger.error(f"HTTP error {response.status} for {method} {endpoint}: {error_text}")
                    self._check_errors_response(response, url=url)

                    if data_to_get == 'json':
                        return await response.json()
                    elif data_to_get == 'text':
                        return await response.text()
                    elif data_to_get == 'content':
                        return await response.read()
                    else:
                        raise ValueError("data_to_get must be either 'json', 'text', or 'content'")

            except aiohttp.ClientError as e:
                logger.error(f"Request error for {method} {endpoint}: {e}")
                raise

        if session is not None:
            return await make_request(session)
        else:
            async with aiohttp.ClientSession() as temp_session:
                return await make_request(temp_session)

    def _make_request_with_pagination(self,
                                      method: str,
                                      endpoint: str,
                                      return_field: str | None = None,
                                      limit: int | None = None,
                                      **kwargs
                                      ) -> Generator[tuple[httpx.Response, list | dict | str], None, None]:
        """Make paginated HTTP requests, yielding each page of results.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            return_field: Optional field name to extract from each item in the response
            limit: Optional maximum number of items to retrieve
            **kwargs: Additional arguments for the request (e.g., params, json)

        Yields:
            Tuples of (HTTP response, items from the current page `response.json()`, for convenience)
        """
        offset = 0
        total_fetched = 0
        params = dict(kwargs.get('params', {}))
        # Ensure kwargs carries our params reference so mutations below take effect
        kwargs['params'] = params

        while True:
            if limit is not None and total_fetched >= limit:
                break

            page_limit = _PAGE_LIMIT
            if limit is not None:
                remaining = limit - total_fetched
                page_limit = min(_PAGE_LIMIT, remaining)

            params['offset'] = offset
            params['limit'] = page_limit

            response = self._make_request(method=method,
                                          endpoint=endpoint,
                                          **kwargs)
            items = self._convert_array_response(response.json(), return_field=return_field)

            if not items:
                break

            items_to_yield = items
            if limit is not None:
                # This ensures we don't yield more than the limit if the API returns more than requested in the last page
                items_to_yield = items[:limit - total_fetched]

            yield response, items_to_yield
            total_fetched += len(items_to_yield)

            if len(items) < _PAGE_LIMIT:
                break

            offset += len(items)

    def _convert_array_response(self,
                                data: dict | list,
                                return_field: str | None = None) -> list | dict | str:
        """Normalize array-like responses into a list when possible.

        Args:
            data: Parsed JSON response.
            return_field: Preferred top-level field to extract when present.

        Returns:
            A list of items when identifiable, otherwise the original data.
        """
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
                 client: httpx.Client | None = None) -> None:
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

    def _make_entity_request(self,
                             method: str,
                             entity_id: str,
                             add_path: str = '',
                             **kwargs) -> httpx.Response:
        try:
            add_path = '/'.join(add_path.strip().strip('/').split('/'))
            return self._make_request(method, f'/{self.endpoint_base}/{entity_id}/{add_path}', **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError(self.endpoint_base, {'id': entity_id}) from e
            raise

    def _stream_entity_request(self,
                               method: str,
                               entity_id: str,
                               add_path: str = '',
                               **kwargs):
        try:
            add_path = '/'.join(add_path.strip().strip('/').split('/'))
            return self._stream_request(method, f'/{self.endpoint_base}/{entity_id}/{add_path}', **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ResourceNotFoundError(self.endpoint_base, {'id': entity_id}) from e
            raise

    def get_list(self, limit: int | None = None, **kwargs) -> Sequence[T]:
        """Get entities with optional filtering.

        Returns:
            List of entity instances.

        Raises:
            httpx.HTTPStatusError: If the request fails.
        """
        params = dict(kwargs)

        # Remove None values from the payload.
        for k in list(params.keys()):
            if params[k] is None:
                del params[k]

        items_gen = self._make_request_with_pagination('GET', f'/{self.endpoint_base}',
                                                       return_field=self.endpoint_base,
                                                       limit=limit,
                                                       params=params)

        all_items = []
        for resp, items in items_gen:
            all_items.extend(items)

        return [self.entity_class(**item) for item in all_items]

    def get_all(self, limit: int | None = None) -> Sequence[T]:
        """Get all entities with optional pagination and filtering.

        Returns:
            List of entity instances

        Raises:
            httpx.HTTPStatusError: If the request fails
        """
        return self.get_list(limit=limit)

    def get_by_id(self, entity_id: str) -> T:
        """Get a specific entity by its ID.

        Args:
            entity_id: Unique identifier for the entity.

        Returns:
            Entity instance.

        Raises:
            httpx.HTTPStatusError: If the entity is not found or request fails.
        """
        response = self._make_entity_request('GET', entity_id)
        return self.entity_class(**response.json())

    def _create(self, entity_data: dict[str, Any]) -> str | list[str | dict]:
        """Create a new entity.

        Args:
            entity_data: Dictionary containing entity data for creation.

        Returns:
            The id of the created entity.

        Raises:
            httpx.HTTPStatusError: If creation fails.
        """
        response = self._make_request('POST', f'/{self.endpoint_base}', json=entity_data)
        respdata = response.json()
        if isinstance(respdata, str):
            return respdata
        if isinstance(respdata, list):
            return respdata
        if isinstance(respdata, dict):
            return respdata.get('id')
        return respdata

    async def _create_async(self, entity_data: dict[str, Any]) -> str | list[str | dict]:
        """Create a new entity.

        Args:
            entity_data: Dictionary containing entity data for creation.

        Returns:
            The id of the created entity.

        Raises:
            httpx.HTTPStatusError: If creation fails.
        """
        respdata = await self._make_request_async('POST',
                                                  f'/{self.endpoint_base}',
                                                  data_to_get='json',
                                                  json=entity_data)
        if isinstance(respdata, str):
            return respdata
        if isinstance(respdata, list):
            return respdata
        if isinstance(respdata, dict):
            return respdata.get('id')
        return respdata

    def update(self, entity_id: str, entity_data: dict[str, Any]) -> T:
        """Update an existing entity.

        Args:
            entity_id: Unique identifier for the entity.
            entity_data: Dictionary containing updated entity data.

        Returns:
            Updated entity instance.

        Raises:
            httpx.HTTPStatusError: If update fails or entity not found.
        """
        response = self._make_entity_request('PUT', entity_id, json=entity_data)
        return self.entity_class(**response.json())

    def delete(self, entity_id: str) -> None:
        """Delete an entity by its ID.

        Args:
            entity_id: Unique identifier for the entity to delete.

        Raises:
            httpx.HTTPStatusError: If deletion fails or entity not found
        """
        self._make_entity_request('DELETE', entity_id)

    def _get_child_entities(self,
                            parent_entity: BaseEntity | str,
                            child_entity_name: str) -> httpx.Response:
        entid = parent_entity if isinstance(parent_entity, str) else parent_entity.id
        # response = self._make_request('GET', f'/{self.endpoint_base}/{entid}/{child_entity_name}')
        response = self._make_entity_request('GET', entid, add_path=child_entity_name)
        return response

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
