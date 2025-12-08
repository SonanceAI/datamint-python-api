import logging
from typing import Any, Generator, AsyncGenerator, Sequence, TYPE_CHECKING
import httpx
from dataclasses import dataclass
from datamint.exceptions import DatamintException, ResourceNotFoundError
import aiohttp
import json
from PIL import Image
import cv2
import nibabel as nib
from io import BytesIO
import gzip
import contextlib
import asyncio
from medimgkit.format_detection import GZIP_MIME_TYPES, DEFAULT_MIME_TYPE, guess_typez, guess_extension

if TYPE_CHECKING:
    from datamint.api.client import Api
    from datamint.types import ImagingData

logger = logging.getLogger(__name__)

_PAGE_LIMIT = 5000


@dataclass
class ApiConfig:
    """Configuration for API client.

    Attributes:
        server_url: Base URL for the API.
        api_key: Optional API key for authentication.
        timeout: Request timeout in seconds.
        max_retries: Maximum number of retries for requests.
        port: Optional port number for the API server.
    """
    server_url: str
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3
    port: int | None = None

    @property
    def web_app_url(self) -> str:
        """Get the base URL for the web application."""
        base_url = self.server_url

        # Add port to base_url if specified
        if self.port is not None:
            base_url = f"{self.server_url.rstrip('/')}:{self.port}"

        if base_url.startswith('http://localhost'):
            return 'http://localhost:3000'
        if base_url.startswith('https://stagingapi.datamint.io'):
            return 'https://staging.datamint.io'
        return 'https://app.datamint.io'


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
        self._owns_client = client is None  # Track if we created the client
        self.client = client or BaseApi._create_client(config)
        self.semaphore = asyncio.Semaphore(20)
        self._api_instance: 'Api | None' = None  # Injected by Api class

    @staticmethod
    def _create_client(config: ApiConfig) -> httpx.Client:
        """Create and configure HTTP client with authentication and timeouts.

        The client is designed to be long-lived and reused across multiple requests.
        It maintains connection pooling for improved performance.
        Default limits: max_keepalive_connections=20, max_connections=100
        """
        headers = {"apikey": config.api_key, 'Authorization': f"Bearer {config.api_key}"} if config.api_key else None

        # Add port to base_url if specified
        base_url = config.server_url.rstrip('/').strip()
        if config.port is not None:
            # if the port is already in the URL, replace it
            if ':' in base_url.split('//')[-1]:
                parts = base_url.rsplit(':', 1)
                # confirm parts[1] is numeric
                if parts[1].isdigit():
                    base_url = f"{parts[0]}:{config.port}"
                else:
                    logger.warning(f"Invalid port detected in server_url: {config.server_url}")
            else:
                base_url = f"{base_url}:{config.port}"

        return httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=config.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5,  # Increased from default 20
                max_connections=20,  # Increased from default 100
                keepalive_expiry=8
            )
        )

    def close(self) -> None:
        """Close the HTTP client and release resources.

        Should be called when the API instance is no longer needed.
        Only closes the client if it was created by this instance.
        """
        if self._owns_client and self.client is not None:
            self.client.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures client is closed."""
        self.close()

    def __del__(self):
        """Destructor - ensures client is closed when instance is garbage collected."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup

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
            if logger.isEnabledFor(logging.DEBUG):
                curl_command = self._generate_curl_command({"method": method,
                                                            "url": url,
                                                            "headers": self.client.headers,
                                                            **kwargs}, fail_silently=True)
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

    def _generate_curl_command(self,
                               request_args: dict,
                               fail_silently: bool = False) -> str:
        """
        Generate a curl command for debugging purposes.

        Args:
            request_args (dict): Request arguments dictionary containing method, url, headers, etc.

        Returns:
            str: Equivalent curl command
        """
        try:
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
        except Exception as e:
            if fail_silently:
                logger.debug(f"Error generating curl command: {e}")
                return "<error generating curl command>"
            raise

    @staticmethod
    def get_status_code(e: httpx.HTTPStatusError | aiohttp.ClientResponseError) -> int:
        if hasattr(e, 'response') and e.response is not None:
            # httpx.HTTPStatusError
            return e.response.status_code
        if hasattr(e, 'status'):
            # aiohttp.ClientResponseError
            return e.status
        if hasattr(e, 'status_code'):
            return e.status_code
        logger.debug(f"Unable to get status code from exception of type {type(e)}")
        return -1

    @staticmethod
    def _has_status_code(e: httpx.HTTPError | aiohttp.ClientResponseError,
                         status_code: int) -> bool:
        return BaseApi.get_status_code(e) == status_code

    def _check_errors_response(self,
                               response: httpx.Response | aiohttp.ClientResponse,
                               url: str):
        try:
            response.raise_for_status()
        except (httpx.HTTPStatusError, aiohttp.ClientResponseError) as e:
            logger.error(f"HTTP error occurred: {e}")
            status_code = BaseApi.get_status_code(e)
            if status_code >= 500 and status_code < 600:
                logger.error(f"Error in request to {url}: {e}")
            if status_code >= 400 and status_code < 500:
                if isinstance(e, aiohttp.ClientResponseError):
                    # aiohttp.ClientResponse does not have .text or .json() methods directly
                    error_msg = e.message
                else:
                    error_msg = e.response.text
                logger.info(f"Error response: {error_msg}")
                if ' not found' in error_msg.lower():
                    # Will be caught by the caller and properly initialized:
                    raise ResourceNotFoundError('unknown', {})
            raise

    @contextlib.asynccontextmanager
    async def _make_request_async(self,
                                  method: str,
                                  endpoint: str,
                                  session: aiohttp.ClientSession | None = None,
                                  **kwargs) -> AsyncGenerator[aiohttp.ClientResponse, None]:
        """Make asynchronous HTTP request with error handling as an async context manager.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            session: Optional aiohttp session. If None, a new one will be created.
            **kwargs: Additional arguments for the request

        Yields:
            An aiohttp.ClientResponse object.

        Raises:
            aiohttp.ClientError: If the request fails

        Example:
            .. code-block:: python

                async with api._make_request_async('GET', '/data') as response:
                    data = await response.json()
        """

        if session is None:
            async with aiohttp.ClientSession() as temp_session:
                async with self._make_request_async(method, endpoint, temp_session, **kwargs) as resp:
                    yield resp
            return

        url = f"{self.config.server_url.rstrip('/')}/{endpoint.lstrip('/')}"

        headers = kwargs.pop('headers', {})
        if self.config.api_key:
            headers['apikey'] = self.config.api_key

        timeout = aiohttp.ClientTimeout(total=self.config.timeout)

        response = None
        curl_cmd = self._generate_curl_command(
            {"method": method, "url": url, "headers": headers, **kwargs},
            fail_silently=True
        )
        logger.debug(f'Equivalent curl command: "{curl_cmd}"')
        async with self.semaphore:
            try:
                response = await session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    timeout=timeout,
                    **kwargs
                )
                self._check_errors_response(response, url=url)
                yield response
            except aiohttp.ClientError as e:
                logger.error(f"Request error for {method} {endpoint}: {e}")
                raise
            finally:
                if response is not None:
                    response.release()

    async def _make_request_async_json(self,
                                       method: str,
                                       endpoint: str,
                                       session: aiohttp.ClientSession | None = None,
                                       **kwargs):
        """Make asynchronous HTTP request and parse JSON response.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path
            session: Optional aiohttp session. If None, a new one will be created.
            **kwargs: Additional arguments for the request

        Returns:
            Parsed JSON response or error information.
        """
        async with self._make_request_async(method, endpoint, session=session, **kwargs) as resp:
            return await resp.json()

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

        use_json_pagination = method.upper() == 'POST' and 'json' in kwargs and isinstance(kwargs['json'], dict)

        if not use_json_pagination:
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

            if use_json_pagination:
                kwargs['json']['offset'] = str(offset)
                kwargs['json']['limit'] = str(page_limit)
            else:
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

    @staticmethod
    def convert_format(bytes_array: bytes,
                       mimetype: str | None = None,
                       file_path: str | None = None
                       ) -> 'ImagingData | bytes':
        """ Convert the bytes array to the appropriate format based on the mimetype.

        Args:
            bytes_array: Raw file content bytes
            mimetype: Optional MIME type of the content
            file_path: deprecated

        Returns:
            Converted content in appropriate format (pydicom.Dataset, PIL Image, cv2.VideoCapture, ...)

        Example:
            >>> fpath = 'path/to/file.dcm'
            >>> with open(fpath, 'rb') as f:
            ...     dicom_bytes = f.read()
            >>> dicom = BaseApi.convert_format(dicom_bytes)

        """
        import pydicom

        if mimetype is None:
            mimetype, ext = BaseApi._determine_mimetype(bytes_array)
            if mimetype is None:
                raise ValueError("Could not determine mimetype from content.")
        content_io = BytesIO(bytes_array)
        if mimetype.endswith('/dicom'):
            return pydicom.dcmread(content_io)
        elif mimetype.startswith('image/'):
            return Image.open(content_io)
        elif mimetype.startswith('video/'):
            if file_path is None:
                raise NotImplementedError("file_path=None is not implemented yet for video/* mimetypes.")
            return cv2.VideoCapture(file_path)
        elif mimetype == 'application/json':
            return json.loads(bytes_array)
        elif mimetype == 'application/octet-stream':
            return bytes_array
        elif mimetype.endswith('nifti'):
            try:
                return nib.Nifti1Image.from_stream(content_io)
            except Exception as e:
                if file_path is not None:
                    return nib.load(file_path)
                raise e
        elif mimetype in GZIP_MIME_TYPES:
            # let's hope it's a .nii.gz
            with gzip.open(content_io, 'rb') as f:
                return nib.Nifti1Image.from_stream(f)

        raise ValueError(f"Unsupported mimetype: {mimetype}")

    @staticmethod
    def _determine_mimetype(content: bytes,
                            declared_mimetype: str | None = None) -> tuple[str | None, str | None]:
        """Infer MIME type and file extension from content and optional declared type.

        Args:
            content: Raw file content bytes
            declared_mimetype: Optional MIME type declared by the source

        Returns:
            Tuple of (inferred_mimetype, file_extension)
        """
        # Determine mimetype from file content
        mimetype_list, ext = guess_typez(content, use_magic=True)
        mimetype = mimetype_list[-1]

        # get mimetype from resource info if not detected
        if declared_mimetype is not None:
            if mimetype is None:
                mimetype = declared_mimetype
                ext = guess_extension(mimetype)
            elif mimetype == DEFAULT_MIME_TYPE:
                mimetype = declared_mimetype
                ext = guess_extension(mimetype)

        return mimetype, ext
