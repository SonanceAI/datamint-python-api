from typing import Optional, Literal, Generator, TypeAlias
import pydicom.dataset
from requests import Session
from requests.exceptions import HTTPError
import logging
import asyncio
import aiohttp
import nest_asyncio  # For running asyncio in jupyter notebooks
import pydicom
import json
from PIL import Image
from io import BytesIO
import cv2
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage as nib_FileBasedImage
from datamint import configs
import gzip
from datamint.exceptions import DatamintException, ResourceNotFoundError
from deprecated.sphinx import deprecated

_LOGGER = logging.getLogger(__name__)


ResourceStatus: TypeAlias = Literal['new', 'inbox', 'published', 'archived']
"""TypeAlias: The available resource status. Possible values: 'new', 'inbox', 'published', 'archived'.
"""
ResourceFields: TypeAlias = Literal['modality', 'created_by', 'published_by', 'published_on', 'filename', 'created_at']
"""TypeAlias: The available fields to order resources. Possible values: 'modality', 'created_by', 'published_by', 'published_on', 'filename', 'created_at' (default).
"""

_PAGE_LIMIT = 5000

@deprecated(reason="Please use `from datamint import Api` instead.", version="2.0.0")
class BaseAPIHandler:
    """
    Class to handle the API requests to the Datamint API
    """
    DATAMINT_API_VENV_NAME = configs.ENV_VARS[configs.APIKEY_KEY]
    DEFAULT_ROOT_URL = 'https://api.datamint.io'

    def __init__(self,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 check_connection: bool = True):
        # deprecated
        _LOGGER.warning("The class APIHandler is deprecated and will be removed in future versions. "
                        "Please use `from datamint import Api` instead.")
        nest_asyncio.apply()  # For running asyncio in jupyter notebooks
        self.root_url = root_url if root_url is not None else configs.get_value(configs.APIURL_KEY)
        if self.root_url is None:
            self.root_url = BaseAPIHandler.DEFAULT_ROOT_URL
        self.root_url.rstrip('/')

        self.api_key = api_key if api_key is not None else configs.get_value(configs.APIKEY_KEY)
        if self.api_key is None:
            msg = f"API key not provided! Use the environment variable " + \
                f"{BaseAPIHandler.DATAMINT_API_VENV_NAME} or pass it as an argument."
            raise DatamintException(msg)
        self.semaphore = asyncio.Semaphore(20)

        if check_connection:
            self.check_connection()

    def check_connection(self):
        try:
            self.get_projects()
        except Exception as e:
            raise DatamintException("Error connecting to the Datamint API." +
                                    f" Please check your api_key and/or other configurations. {e}")

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

    async def _run_request_async(self,
                                 request_args: dict,
                                 session: aiohttp.ClientSession | None = None,
                                 data_to_get: Literal['json', 'text', 'content'] = 'json'):
        if session is None:
            async with aiohttp.ClientSession() as s:
                return await self._run_request_async(request_args, s, data_to_get)

        async with self.semaphore:
            try:
                _LOGGER.debug(f"Running request to {request_args['url']}")
                _LOGGER.debug(f'Equivalent curl command: "{self._generate_curl_command(request_args)}"')
            except Exception as e:
                _LOGGER.debug(f"Error generating curl command: {e}")

            # add apikey to the headers
            if 'headers' not in request_args:
                request_args['headers'] = {}

            request_args['headers']['apikey'] = self.api_key

            async with session.request(**request_args) as response:
                self._check_errors_response(response, request_args)
                if data_to_get == 'json':
                    return await response.json()
                elif data_to_get == 'text':
                    return await response.text()
                elif data_to_get == 'content':
                    return await response.read()
                else:
                    raise ValueError("data_to_get must be either 'json' or 'text'")

    def _check_errors_response(self,
                               response,
                               request_args: dict):
        try:
            response.raise_for_status()
        except HTTPError as e:
            status_code = BaseAPIHandler.get_status_code(e)
            if status_code >= 500 and status_code < 600:
                _LOGGER.error(f"Error in request to {request_args['url']}: {e}")
            if status_code >= 400 and status_code < 500:
                try:
                    _LOGGER.info(f"Error response: {response.text}")
                    error_data = response.json()
                except Exception as e2:
                    _LOGGER.info(f"Error parsing the response. {e2}")
                else:
                    if isinstance(error_data['message'], str) and ' not found' in error_data['message'].lower():
                        # Will be caught by the caller and properly initialized:
                        raise ResourceNotFoundError('unknown', {})

            raise e

    def _check_errors_response_json(self,
                                    response):
        response_json = response.json()
        if isinstance(response_json, dict):
            response_json = [response_json]
        if isinstance(response_json, list):
            for r in response_json:
                if isinstance(r, dict) and 'error' in r:
                    if hasattr(response, 'text'):
                        _LOGGER.error(f"Error response: {response.text}")
                    raise DatamintException(r['error'])

    def _run_request(self,
                     request_args: dict,
                     session: Session | None = None):
        if session is None:
            with Session() as s:
                return self._run_request(request_args, s)
        _LOGGER.debug(f'Equivalent curl command: "{self._generate_curl_command(request_args)}"')

        # add apikey to the headers
        if 'headers' not in request_args:
            request_args['headers'] = {}

        request_args['headers']['apikey'] = self.api_key
        response = session.request(**request_args)
        self._check_errors_response(response, request_args)
        return response

    def _get_endpoint_url(self, endpoint: str) -> str:
        return f'{self.root_url}/{endpoint}'

    def _run_pagination_request(self,
                                request_params: dict,
                                return_field: str | list | None = None
                                ) -> Generator[dict | list, None, None]:
        offset = 0
        params = request_params.get('params', {})
        while True:
            params['offset'] = offset
            params['limit'] = _PAGE_LIMIT

            response = self._run_request(request_params)
            self._check_errors_response_json(response)
            response = response.json()
            if return_field is not None:
                if isinstance(return_field, list) or isinstance(return_field, tuple):
                    for field in return_field:
                        response = response[field]
                else:
                    response = response[return_field]
            for r in response:
                yield r

            if len(response) < _PAGE_LIMIT:
                _LOGGER.debug(f"Last page reached. Total resources: {offset + len(response)}")
                break

            offset += _PAGE_LIMIT

    @staticmethod
    def get_status_code(e) -> int:
        if not hasattr(e, 'response') or e.response is None:
            return -1
        return e.response.status_code

    @staticmethod
    def _has_status_code(e, status_code: int) -> bool:
        return BaseAPIHandler.get_status_code(e) == status_code

    @staticmethod
    def convert_format(bytes_array: bytes,
                       mimetype: str,
                       file_path: str | None = None
                       ) -> pydicom.dataset.Dataset | Image.Image | cv2.VideoCapture | bytes | nib_FileBasedImage:
        """ Convert the bytes array to the appropriate format based on the mimetype."""
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
        elif mimetype == 'application/gzip':
            # let's hope it's a .nii.gz
            with gzip.open(content_io, 'rb') as f:
                return nib.Nifti1Image.from_stream(f)

        raise ValueError(f"Unsupported mimetype: {mimetype}")
