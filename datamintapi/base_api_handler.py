from typing import Optional, IO, Sequence, Literal, Generator, TypeAlias, Dict, Tuple, Union, List
import os
import pydicom.dataset
from requests import Session
from requests.exceptions import HTTPError
import logging
import asyncio
import aiohttp
import nest_asyncio  # For running asyncio in jupyter notebooks
from datamintapi.utils.dicom_utils import anonymize_dicom, to_bytesio, is_dicom
import pydicom
from pathlib import Path
from datetime import date
import mimetypes
import json
from PIL import Image
from io import BytesIO
import cv2
import nibabel as nib
from nibabel.filebasedimages import FileBasedImage as nib_FileBasedImage
from deprecated.sphinx import deprecated
import pydantic
from datamintapi import configs
import numpy as np

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


ResourceStatus: TypeAlias = Literal['new', 'inbox', 'published', 'archived']
"""TypeAlias: The available resource status. Possible values: 'new', 'inbox', 'published', 'archived'.
"""
ResourceFields: TypeAlias = Literal['modality', 'created_by', 'published_by', 'published_on', 'filename']
"""TypeAlias: The available fields to order resources. Possible values: 'modality', 'created_by', 'published_by', 'published_on', 'filename'.
"""

_PAGE_LIMIT = 10


def validate_call(func, *args, **kwargs):
    """
    wraps the function with pydantic's validate_call decorator to only warn about validation errors.
    """
    new_func = pydantic.validate_call(func, *args, **kwargs)

    def wrapper(*args, **kwargs):
        try:
            return new_func(*args, **kwargs)
        except pydantic.ValidationError as e:
            _LOGGER.warning(f"Validation error: {e}")
            return func(*args, **kwargs)

    return wrapper


class DatamintException(Exception):
    """
    Base class for exceptions in this module.
    """
    pass


class ResourceNotFoundError(DatamintException):
    """
    Exception raised when a resource is not found. 
    For instance, when trying to get a resource by a non-existing id.
    """

    def __init__(self,
                 resource_type: str,
                 params: dict):
        """ Constructor.

        Args:
            resource_type (str): A resource type.
            params (dict): Dict of params identifying the sought resource.
        """
        super().__init__()
        self.resource_type = resource_type
        self.params = params

    def set_params(self, resource_type:str, params: dict):
        self.resource_type = resource_type
        self.params = params

    def __str__(self):
        return f"Resource '{self.resource_type}' not found for parameters: {self.params}"


class BaseAPIHandler:
    """
    Class to handle the API requests to the Datamint API
    """
    DATAMINT_API_VENV_NAME = configs.ENV_VARS[configs.APIKEY_KEY]
    DEFAULT_ROOT_URL = 'https://api.datamint.io'

    def __init__(self,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        nest_asyncio.apply()  # For running asyncio in jupyter notebooks
        self.root_url = root_url if root_url is not None else configs.get_value(configs.APIURL_KEY)
        if self.root_url is None:
            self.root_url = BaseAPIHandler.DEFAULT_ROOT_URL

        self.api_key = api_key if api_key is not None else configs.get_value(configs.APIKEY_KEY)
        if self.api_key is None:
            msg = f"API key not provided! Use the environment variable " + \
                f"{BaseAPIHandler.DATAMINT_API_VENV_NAME} or pass it as an argument."
            raise DatamintException(msg)
        self.semaphore = asyncio.Semaphore(10)  # Limit to 10 parallel requests

    async def _run_request_async(self,
                                 request_args: dict,
                                 session: aiohttp.ClientSession = None,
                                 data_to_get: str = 'json'):
        if session is None:
            async with aiohttp.ClientSession() as s:
                return await self._run_request_async(request_args, s)

        _LOGGER.debug(f"Running request to {request_args['url']}")
        _LOGGER.debug(f"Request args: {request_args}")

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
                    _LOGGER.error(f"Error response: {response.text}")
                    error_data = response.json()
                except Exception as e2:
                    _LOGGER.error(f"Error parsing the response. {e2}")
                else:
                    if ' not found' in error_data['message'].lower():
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
                     session: Session = None):
        if session is None:
            with Session() as s:
                return self._run_request(request_args, s)

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
                                request_params: Dict,
                                return_field: Optional[Union[str, List]] = None
                                ) -> Generator[Dict, None, None]:
        offset = 0
        params = request_params['params']
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
                       file_path: str = None
                       ) -> pydicom.dataset.Dataset | Image.Image | cv2.VideoCapture | bytes | nib_FileBasedImage:
        content_io = BytesIO(bytes_array)
        if mimetype == 'application/dicom':
            return pydicom.dcmread(content_io)
        elif mimetype in ('image/jpeg', 'image/png', 'image/tiff'):
            return Image.open(content_io)
        elif mimetype == 'video/mp4':
            if file_path is None:
                raise NotImplementedError("file_path=None is not implemented yet for video/mp4.")
            return cv2.VideoCapture(file_path)
        elif mimetype == 'application/json':
            return json.loads(bytes_array)
        elif mimetype == 'application/octet-stream':
            return bytes_array
        elif mimetype == 'application/nifti':
            if file_path is None:
                raise NotImplementedError("file_path=None is not implemented yet for application/nifti.")
            return nib.load(file_path)

        raise ValueError(f"Unsupported mimetype: {mimetype}")

    def create_project(self,
                       name: str,
                       description: str,
                       is_active_learning: bool = False) -> dict:
        """
        Create a new project.

        Args:
            name (str): The name of the project.

        Returns:
            dict: The created project.

        Raises:
            DatamintException: If the project could not be created.
        """
        request_args = {
            'url': self._get_endpoint_url('projects'),
            'method': 'POST',
            'json': {'name': name,
                     'is_active_learning': is_active_learning,
                     'description': description}
        }
        response = self._run_request(request_args)
        self._check_errors_response_json(response)
        return response.json()
