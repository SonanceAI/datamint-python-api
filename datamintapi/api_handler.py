from typing import Optional, IO, Sequence, Literal, Generator, TypeAlias
import os
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
from datamintapi import configs

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')

ResourceStatus: TypeAlias = Literal['new', 'inbox', 'published', 'archived']
"""TypeAlias: The available resource status. Possible values: 'new', 'inbox', 'published', 'archived'.
"""
ResourceFields: TypeAlias = Literal['modality', 'created_by', 'published_by', 'published_on', 'filename']
"""TypeAlias: The available fields to order resources. Possible values: 'modality', 'created_by', 'published_by', 'published_on', 'filename'.
"""

_PAGE_LIMIT = 10


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
        super().__init__(f"Resource '{resource_type}' not found for parameters: {params}")
        self.resource_type = resource_type
        self.params = params


def _is_io_object(obj):
    """
    Check if an object is a file-like object.
    """
    return callable(getattr(obj, "read", None))


def _open_io(file_path: str | Path | IO, mode: str = 'rb') -> IO:
    if isinstance(file_path, str) or isinstance(file_path, Path):
        return open(file_path, 'rb')
    return file_path


class APIHandler:
    """
    Class to handle the API requests to the Datamint API
    """
    DATAMINT_API_VENV_NAME = configs.ENV_VARS[configs.APIKEY_KEY]
    ENDPOINT_RESOURCES = 'resources'
    ENDPOINT_DICOMS = 'dicoms'
    DEFAULT_ROOT_URL = 'https://stagingapi.datamint.io'

    def __init__(self,
                 root_url: Optional[str] = None,
                 api_key: Optional[str] = None):
        nest_asyncio.apply()  # For running asyncio in jupyter notebooks
        self.root_url = root_url if root_url is not None else configs.get_value(configs.APIURL_KEY)
        if self.root_url is None:
            self.root_url = APIHandler.DEFAULT_ROOT_URL

        self.api_key = api_key if api_key is not None else configs.get_value(configs.APIKEY_KEY)
        if self.api_key is None:
            msg = f"API key not provided! Use the environment variable " + \
                f"{APIHandler.DATAMINT_API_VENV_NAME} or pass it as an argument."
            raise DatamintException(msg)
        self.semaphore = asyncio.Semaphore(10)  # Limit to 10 parallel requests

    async def _run_request_async(self,
                                 request_args: dict,
                                 session: aiohttp.ClientSession = None,
                                 data_to_get: str = 'json'):
        if session is None:
            async with aiohttp.ClientSession() as s:
                return await self._run_request_async(request_args, s)

        _LOGGER.info(f"Running request to {request_args['url']}")
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
            if response.status_code == 404:
                raise DatamintException(f'The requested endpoint "{request_args["url"]}" could not be found. ' +
                                        'Please check the URL and try again')
            raise e

    def _run_request(self,
                     request_args: dict,
                     session=None):
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

    def create_batch(self,
                     description: str,
                     size: int,
                     modality: Optional[str] = None,
                     session=None) -> str:
        """
        Create a new batch.

        Args:
            description (str): The description of the batch
            size (int): The number of dicoms in the batch
            modality (str, optional): The modality of the batch. Defaults to None.

        Returns:
            str: The batch_id of the created batch. 

        See Also:
            :meth:`~get_batch_info`
        """
        post_params = {'description': description,
                       'size': size
                       }
        if modality is not None:
            post_params['modality'] = modality

        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/upload-batches',
            'json': post_params
        }

        resp = self._run_request(request_params, session)
        return resp.json()['id']

    def _get_endpoint_url(self, endpoint: str) -> str:
        return f'{self.root_url}/{endpoint}'

    async def _upload_single_resource_async(self,
                                            file_path: str | IO,
                                            batch_id: Optional[str] = None,
                                            anonymize: bool = False,
                                            anonymize_retain_codes: Sequence[tuple] = [],
                                            labels: list[str] = None,
                                            mung_filename: Sequence[int] | Literal['all'] = None,
                                            channel: Optional[str] = None,
                                            session=None,
                                            modality: Optional[str] = None,
                                            ) -> str:
        if _is_io_object(file_path):
            name = file_path.name
        else:
            name = file_path

        if session is not None and not isinstance(session, aiohttp.ClientSession):
            raise ValueError("session must be an aiohttp.ClientSession object.")

        name = os.path.expanduser(os.path.normpath(name))
        name = os.path.join(*[x if x != '..' else '_' for x in Path(name).parts])

        if mung_filename is not None:
            file_parts = Path(name).parts
            if file_parts[0] == os.path.sep:
                file_parts = file_parts[1:]
            if mung_filename == 'all':
                new_file_path = '_'.join(file_parts)
            else:
                folder_parts = file_parts[:-1]
                new_file_path = '_'.join([folder_parts[i-1] for i in mung_filename if i <= len(folder_parts)])
                new_file_path += '_' + file_parts[-1]
            name = new_file_path
            _LOGGER.debug(f"New file path: {name}")

        mimetype = mimetypes.guess_type(name)[0]
        is_a_dicom_file = None
        if mimetype is None:
            is_a_dicom_file = is_dicom(name) or is_dicom(file_path)
            if is_a_dicom_file:
                mimetype = 'application/dicom'

        _LOGGER.debug(f"File name '{name}' mimetype: {mimetype}")

        if anonymize:
            if is_a_dicom_file == True or is_dicom(file_path):
                ds = pydicom.dcmread(file_path)
                ds = anonymize_dicom(ds, retain_codes=anonymize_retain_codes)
                # make the dicom `ds` object a file-like object in order to avoid unnecessary disk writes
                f = to_bytesio(ds, name)
            else:
                _LOGGER.warning(f"File {file_path} is not a dicom file. Skipping anonymization.")
                f = _open_io(file_path)
        else:
            f = _open_io(file_path)

        try:
            form = aiohttp.FormData()
            if batch_id is not None:
                url = self._get_endpoint_url(APIHandler.ENDPOINT_DICOMS)
                file_key = 'dicom'
                if modality is not None:
                    _LOGGER.warning("Modality is ignored when uploading to a batch.")
                modality = None
                form.add_field('batch_id', batch_id)
            else:
                url = self._get_endpoint_url(APIHandler.ENDPOINT_RESOURCES)
                file_key = 'resource'
                form.add_field('source', 'api')

            form.add_field(file_key, f, filename=name, content_type=mimetype)
            form.add_field('filepath', name)
            if mimetype is not None:
                form.add_field('mimetype', mimetype)
            if channel is not None:
                form.add_field('channel', channel)
            if modality is not None:
                form.add_field('modality', modality)
            if labels is not None:
                for i, label in enumerate(labels):
                    form.add_field(f'labels[{i}]', label)

            request_params = {
                'method': 'POST',
                'url': url,
                'data': form,
            }

            resp_data = await self._run_request_async(request_params, session)
            if 'error' in resp_data:
                raise DatamintException(resp_data['error'])
            _LOGGER.info(f"Response on uploading {file_path}: {resp_data}")

            _USER_LOGGER.info(f'{file_path} uploaded')
            return resp_data['id']
        except Exception as e:
            _LOGGER.error(f"Error uploading {file_path}: {e}")
            raise e
        finally:
            f.close()

    async def _upload_resources_async(self,
                                      files_path: Sequence[str | IO],
                                      batch_id: Optional[str] = None,
                                      anonymize: bool = False,
                                      anonymize_retain_codes: Sequence[tuple] = [],
                                      on_error: Literal['raise', 'skip'] = 'raise',
                                      labels=None,
                                      mung_filename: Sequence[int] | Literal['all'] = None,
                                      channel: Optional[str] = None,
                                      modality: Optional[str] = None,
                                      ) -> list[str]:
        if on_error not in ['raise', 'skip']:
            raise ValueError("on_error must be either 'raise' or 'skip'")

        async with aiohttp.ClientSession() as session:

            async def __upload_single_resource(file_path):
                _LOGGER.debug(f"Current semaphore value: {self.semaphore._value}")
                async with self.semaphore:
                    return await self._upload_single_resource_async(
                        file_path=file_path,
                        anonymize=anonymize,
                        anonymize_retain_codes=anonymize_retain_codes,
                        labels=labels,
                        session=session,
                        mung_filename=mung_filename,
                        channel=channel,
                        modality=modality,
                        batch_id=batch_id
                    )
            tasks = [__upload_single_resource(f) for f in files_path]
            return await asyncio.gather(*tasks, return_exceptions=on_error == 'skip')

    def upload_dicoms(self,
                      files_path: str | IO | Sequence[str | IO],
                      batch_id: Optional[str] = None,
                      anonymize: bool = False,
                      anonymize_retain_codes: Sequence[tuple] = [],
                      on_error: Literal['raise', 'skip'] = 'raise',
                      labels=None,
                      mung_filename: Sequence[int] | Literal['all'] = None,
                      channel: Optional[str] = None,
                      ) -> list[str]:
        """
        Upload dicoms. If batch_id is None, this is equivalent to :meth:`~upload_resources`.

        Args:
            files_path (str | IO | Sequence[str | IO]): The path to the dicom file or a list of paths to dicom files.
            batch_id (Optional[str]): The batch unique id. If None, the dicom will be uploaded to the resources tab.
            anonymize (bool): Whether to anonymize the dicoms or not.
            anonymize_retain_codes (Sequence[tuple]): The tags to retain when anonymizing the dicoms.
            on_error (Literal['raise', 'skip']): Whether to raise an exception when an error occurs or to skip the error.
            labels (list[str]): The labels to assign to the dicoms.
            mung_filename (Sequence[int] | Literal['all']): The parts of the filepath to keep when renaming the dicom file.
                ''all'' keeps all parts.
            channel (Optional[str]): The channel to upload the dicoms to. An arbitrary name to group the dicoms.

        Returns:
            list[str]: The list of new created dicom_ids.

        Raises:
            ResourceNotFoundError: If the batch does not exists.
        """
        files_path = APIHandler.__process_files_parameter(files_path)
        loop = asyncio.get_event_loop()
        task = self._upload_resources_async(files_path=files_path,
                                            batch_id=batch_id,
                                            anonymize=anonymize,
                                            anonymize_retain_codes=anonymize_retain_codes,
                                            on_error=on_error,
                                            labels=labels,
                                            mung_filename=mung_filename,
                                            channel=channel,
                                            )

        return loop.run_until_complete(task)

    def upload_resources(self,
                         files_path: str | IO | Sequence[str | IO],
                         anonymize: bool = False,
                         anonymize_retain_codes: Sequence[tuple] = [],
                         on_error: Literal['raise', 'skip'] = 'raise',
                         labels=None,
                         mung_filename: Sequence[int] | Literal['all'] = None,
                         channel: Optional[str] = None,
                         ) -> list[str]:
        """
        Upload resources.

        Args:
            files_path (str | IO | Sequence[str | IO]): The path to the dicom file or a list of paths to dicom files.
            anonymize (bool): Whether to anonymize the dicoms or not.
            anonymize_retain_codes (Sequence[tuple]): The tags to retain when anonymizing the dicoms.
            on_error (Literal['raise', 'skip']): Whether to raise an exception when an error occurs or to skip the error.
            labels (list[str]): The labels to assign to the dicoms.
            mung_filename (Sequence[int] | Literal['all']): The parts of the filepath to keep when renaming the dicom file.
                ''all'' keeps all parts.
            channel (Optional[str]): The channel to upload the dicoms to. An arbitrary name to group the dicoms.

        Returns:
            list[str]: The list of new created dicom_ids.
        """
        files_path = APIHandler.__process_files_parameter(files_path)
        loop = asyncio.get_event_loop()
        task = self._upload_resources_async(files_path=files_path,
                                            anonymize=anonymize,
                                            anonymize_retain_codes=anonymize_retain_codes,
                                            on_error=on_error,
                                            labels=labels,
                                            mung_filename=mung_filename,
                                            channel=channel,
                                            )

        return loop.run_until_complete(task)

    @staticmethod
    def __process_files_parameter(file_path: str | IO | Sequence[str | IO]) -> Sequence[str | IO]:
        if isinstance(file_path, str):
            if os.path.isdir(file_path):
                file_path = [f'{file_path}/{f}' for f in os.listdir(file_path)]
            else:
                file_path = [file_path]
        # Check if is an IO object
        elif _is_io_object(file_path):
            file_path = [file_path]
        elif not hasattr(file_path, '__len__'):
            if hasattr(file_path, '__iter__'):
                file_path = list(file_path)
            else:
                file_path = [file_path]

        _LOGGER.debug(f'Processed file path: {file_path}')
        return file_path

    def create_batch_with_dicoms(self,
                                 description: str,
                                 files_path: str | IO | Sequence[str | IO],
                                 on_error: Literal['raise', 'skip'] = 'raise',
                                 labels: Sequence[str] = None,
                                 modality: Optional[str] = None,
                                 anonymize: bool = False,
                                 anonymize_retain_codes: Sequence[tuple] = [],
                                 mung_filename: Sequence[int] | Literal['all'] = None,
                                 channel: Optional[str] = None,
                                 ) -> tuple[str, list[str]]:
        """
        Handy method to create a new batch and upload the dicoms in a single call.

        Args:
            description (str): The description of the batch
            files_path (str | IO | Sequence): The path to the dicom file or a list of paths to dicom files.
            on_error (Literal['raise', 'skip']): Whether to raise an exception when an error occurs or to skip the error.
            labels (Sequence[str]): The labels to assign to the dicoms.
            modality (str, optional): The modality of the batch. Defaults to None.
            anonymize (bool): Whether to anonymize the dicoms or not.
            anonymize_retain_codes (Sequence[tuple]): The tags to retain when anonymizing the dicoms.
            mung_filename (Sequence[int] | Literal['all']): The parts of the filepath to keep when renaming the dicom file.
                ''all'' keeps all parts.

        Returns:
            tuple[str, list[str]]: The batch_id and the list of new created dicom_ids.

        Examples:
            .. tabs::
                .. tab:: Example 1

                    >>> api_handler.create_batch_with_dicoms('New batch', 'path/to/dicom.dcm')

                .. tab:: Example 2

                    >>> with open('path/to/dicom.dcm', 'rb') as f:
                    >>>     api_handler.create_batch_with_dicoms('New batch', f)

                .. tab:: Example 3    

                    >>> api_handler.create_batch_with_dicoms('New batch', ['path/to/dicom1.dcm', 'path/to/dicom2.dcm'])

                .. tab:: Example 4

                    >>> api_handler.create_batch_with_dicoms('New batch', 'path/to/dicom.dcm', anonymize=True)
                    >>> api_handler.create_batch_with_dicoms('New batch', 'path/to/dicom.dcm', anonymize=True, anonymize_retain_codes=[(0x0010, 0x0010)])

                .. tab:: Example 5

                    >>> api_handler.create_batch_with_dicoms('New batch', 'path/to/dicom.dcm', mung_filename=[1, 2, 3])
                    >>> api_handler.create_batch_with_dicoms('New batch', 'path/to/dicom.dcm', mung_filename='all')

        """

        files_path = APIHandler.__process_files_parameter(files_path)

        if labels is not None:
            labels = [l.strip() for l in labels]

        batch_id = self.create_batch(description,
                                     size=len(files_path),
                                     modality=modality)
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._upload_resources_async(files_path,
                                                                       batch_id=batch_id,
                                                                       anonymize=anonymize,
                                                                       anonymize_retain_codes=anonymize_retain_codes,
                                                                       on_error=on_error,
                                                                       labels=labels,
                                                                       channel=channel,
                                                                       mung_filename=mung_filename)
                                          )
        return batch_id, results

    def get_batches(self) -> Generator[dict, None, None]:
        """
        Iterate over all the batches.

        Returns:
            Generator[dict, None, None]: A generator of dictionaries with information about the batches.

        Example:
            >>> for batch in api_handler.get_batches():
            >>>     print(batch)
        """

        offset = 0
        limit_page = 10

        request_params = {
            'method': 'GET',
            'params': {'offset': offset, 'limit': limit_page},
            'url': f'{self.root_url}/upload-batches'
        }

        results = self._run_request(request_params).json()['data']
        while len(results) > 0:
            for result in results:
                yield result
            if len(results) < limit_page:
                break
            offset += limit_page
            request_params['params']['offset'] = offset
            results = self._run_request(request_params).json()['data']

    def get_batch_info(self, batch_id: str) -> dict:
        """
        Get information of a batch.

        Args:
            batch_id (str): The batch unique id.

        Returns:
            dict: Informations the batch and its images.

        Raises:
            ResourceNotFoundError: If the batch does not exists.

        """
        try:
            request_params = {
                'method': 'GET',
                'url': f'{self.root_url}/upload-batches/{batch_id}'
            }
            return self._run_request(request_params).json()
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ResourceNotFoundError('batch', {'batch_id': batch_id})
            raise e

    def upload_segmentation(self,
                            dicom_id: str,
                            file_path: str,
                            segmentation_name: str,
                            task_id: Optional[str] = None,
                            ) -> str:
        """
        Upload a segmentation to a dicom.

        Args:
            dicom_id (str): The dicom unique id.
            file_path (str): The path to the segmentation file.
            segmentation_name (str): The segmentation name.
            task_id (Optional[str]): The task unique id.

        Returns:
            str: The segmentation unique id.

        Raises:
            ResourceNotFoundError: If the dicom does not exists or the segmentation is invalid.

        Example:
            >>> batch_id, dicoms_ids = api_handler.create_batch_with_dicoms('New batch', 'path/to/dicom.dcm')
            >>> api_handler.upload_segmentation(dicoms_ids[0], 'path/to/segmentation.nifti', 'Segmentation name')
        """
        try:
            with open(file_path, 'rb') as f:
                request_params = dict(
                    method='POST',
                    url=self.root_url+'/segmentations',
                    data={'dicomId': dicom_id,
                          'segmentationName': [segmentation_name],
                          },
                    files={'segmentationData': f}
                )
                if task_id is not None:
                    request_params['data']['taskId'] = task_id
                return self._run_request(request_params).json()['id']
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ResourceNotFoundError('dicom', {'dicom_id': dicom_id})
            raise e

    def get_resources_by_ids(self, ids: str | Sequence[str]) -> dict | Sequence[dict]:
        """
        Get resources by their unique ids.

        Args:
            ids (str | Sequence[str]): The resource unique id or a list of resource unique ids.

        Returns:
            dict | Sequence[dict]: The resource information or a list of resource information.

        Raises:
            ResourceNotFoundError: If the resource does not exists.

        Example:
            >>> api_handler.get_resources_by_ids('resource_id')
            >>> api_handler.get_resources_by_ids(['resource_id1', 'resource_id2'])
        """
        input_is_a_string = isinstance(ids, str)  # used later to return a single object or a list of objects
        if input_is_a_string:
            ids = [ids]

        resources = []
        try:
            for i in ids:
                request_params = {
                    'method': 'GET',
                    'url': f'{self.root_url}/resources/{i}',
                }

                resources.append(self._run_request(request_params).json())
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ResourceNotFoundError('resource', {'resource_id': i})
            raise e

        return resources[0] if input_is_a_string else resources

    def get_resources(self,
                      status: ResourceStatus,
                      from_date: Optional[date] = None,
                      to_date: Optional[date] = None,
                      labels: Optional[list[str]] = None,
                      modality: Optional[str] = None,
                      mimetype: Optional[str] = None,
                      return_ids_only: bool = False,
                      order_field: Optional[ResourceFields] = None,
                      order_ascending: Optional[bool] = None,
                      ) -> Generator[dict, None, None]:
        """
        Iterates over resources with the specified filters.
        Filters can be combined to narrow down the search.
        It returns full information of the resources by default, but it can be configured to return only the ids with parameter `return_ids_only`.

        Args:
            status (ResourceStatus): The resource status. Possible values: 'inbox', 'published' or 'archived'.
            from_date (Optional[date]): The start date.
            to_date (Optional[date]): The end date.
            labels (Optional[list[str]]): The labels to filter the resources.
            modality (Optional[str]): The modality of the resources.
            mimetype (Optional[str]): The mimetype of the resources.
            return_ids_only (bool): Whether to return only the ids of the resources.
            order_field (Optional[ResourceFields]): The field to order the resources. See :data:`~ResourceFields`.
            order_ascending (Optional[bool]): Whether to order the resources in ascending order.

        Returns:
            Generator[dict, None, None]: A generator of dictionaries with the resources information.

        Example:
            >>> for resource in api_handler.get_resources(status='inbox'):
            >>>     print(resource)
        """
        # Convert datetime objects to ISO format
        if from_date:
            from_date = from_date.isoformat()
        if to_date:
            to_date = to_date.isoformat()

        # Prepare the payload
        payload = {
            "from": from_date,
            "to": to_date,
            "modality": modality,
            "status": status,
            "mimetype": mimetype,
            "ids": return_ids_only,
            "order_field": order_field,
            "order_by_asc": order_ascending,
        }
        if labels is not None:
            for i, label in enumerate(labels):
                payload[f'labels[{i}]'] = label

        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/resources',
            'params': payload
        }

        offset = 0
        while True:
            payload['offset'] = offset
            payload['limit'] = _PAGE_LIMIT

            response = self._run_request(request_params).json()

            for r in response:
                yield r

            if len(response) < _PAGE_LIMIT:
                break

            offset += _PAGE_LIMIT
