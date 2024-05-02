from typing import Optional, IO, Sequence, Literal
import os
from requests import Session
import logging
import asyncio
import aiohttp
import nest_asyncio  # For running asyncio in jupyter notebooks
from datamintapi.utils.dicom_utils import anonymize_dicom, to_bytesio
import pydicom
from pathlib import Path

_LOGGER = logging.getLogger(__name__)
_USER_LOGGER = logging.getLogger('user_logger')


class DatamintException(Exception):
    pass


class DicomAlreadyStored(DatamintException):
    pass


class APIHandler:
    """
    Class to handle the API requests to the Datamint API
    """
    DATAMINT_API_VENV_NAME = 'DATAMINT_API_KEY'

    def __init__(self,
                 root_url: str,
                 api_key: Optional[str] = None):
        nest_asyncio.apply()  # For running asyncio in jupyter notebooks
        self.root_url = root_url
        self.api_key = api_key if api_key is not None else os.getenv(APIHandler.DATAMINT_API_VENV_NAME)
        if self.api_key is None:
            msg = f"API key not provided! Use the environment variable {APIHandler.DATAMINT_API_VENV_NAME} or pass it as an argument."
            raise Exception(msg)

    async def _run_request_async(self,
                                 request_args: dict,
                                 session=None,
                                 data_to_get: str = 'json'):
        if session is None:
            async with aiohttp.ClientSession() as s:
                return await self._run_request_async(request_args, s)

        _LOGGER.info(f"Running request to {request_args['url']}")

        # add apikey to the headers
        if 'headers' not in request_args:
            request_args['headers'] = {}

        request_args['headers']['apikey'] = self.api_key
        async with session.request(**request_args) as response:
            response.raise_for_status()
            if data_to_get == 'json':
                return await response.json()
            elif data_to_get == 'text':
                return await response.text()
            else:
                raise ValueError("data_to_get must be either 'json' or 'text'")

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
        response.raise_for_status()
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

    async def _upload_single_dicom_async(self, batch_id: str,
                                         file_path: str | IO,
                                         anonymize: bool = False,
                                         anonymize_retain_codes: Sequence[tuple] = [],
                                         labels: list[str] = None,
                                         mung_filename: Sequence[int] | Literal['all'] = None,
                                         session=None) -> str:
        if anonymize:
            ds = pydicom.dcmread(file_path)
            ds = anonymize_dicom(ds, retain_codes=anonymize_retain_codes)

            if mung_filename is not None:
                file_parts = Path(file_path).parts
                if file_parts[0] == os.path.sep:
                    file_parts = file_parts[1:]
                if mung_filename == 'all':
                    new_file_path = '_'.join(file_parts)
                else:
                    folder_parts = file_parts[:-1]
                    new_file_path = '_'.join([folder_parts[i-1] for i in mung_filename if i <= len(folder_parts)])
                    new_file_path += '_' + file_parts[-1]
                name = new_file_path
                _LOGGER.debug(f"New file path: {new_file_path}")
            else:
                name = file_path
            # make the dicom `ds` object a file-like object in order to avoid unnecessary disk writes
            f = to_bytesio(ds, name)
        elif isinstance(file_path, str):
            f = open(file_path, 'rb')
        else:
            f = file_path

        try:
            request_params = {
                'method': 'POST',
                'url': f'{self.root_url}/dicoms',
                'data': {'batch_id': batch_id, 'dicom': f, 'filepath': file_path}
            }

            if labels is not None:
                request_params['data']['labels[]'] = ','.join(labels)
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

    async def _upload_dicoms_async(self,
                                   files_path: Sequence[str | IO],
                                   batch_id: str,
                                   anonymize: bool = False,
                                   anonymize_retain_codes: Sequence[tuple] = [],
                                   on_error: Literal['raise', 'skip'] = 'raise',
                                   labels=None,
                                   mung_filename: Sequence[int] | Literal['all'] = None,
                                   ) -> list[str]:
        if on_error not in ['raise', 'skip']:
            raise ValueError("on_error must be either 'raise' or 'skip'")

        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(10)  # Limit to 10 parallel requests

            async def __upload_single_dicom(file_path):
                _LOGGER.debug(f"Current semaphore value: {semaphore._value}")
                async with semaphore:
                    return await self._upload_single_dicom_async(
                        batch_id, file_path, anonymize, anonymize_retain_codes,
                        labels=labels,
                        session=session,
                        mung_filename=mung_filename
                    )
            tasks = [__upload_single_dicom(f) for f in files_path]
            return await asyncio.gather(*tasks, return_exceptions=on_error == 'skip')

    def upload_dicoms(self,
                      files_path: str | IO | Sequence[str | IO],
                      batch_id: str,
                      anonymize: bool = False,
                      anonymize_retain_codes: Sequence[tuple] = [],
                      on_error: Literal['raise', 'skip'] = 'raise',
                      labels=None,
                      mung_filename: Sequence[int] | Literal['all'] = None,
                      ) -> list[str]:
        """
        Upload dicoms to a batch.

        Args:
            files_path (str | IO | Sequence[str | IO]): The path to the dicom file or a list of paths to dicom files.
            batch_id (str): The batch unique id.
            anonymize (bool): Whether to anonymize the dicoms or not.
            anonymize_retain_codes (Sequence[tuple]): The tags to retain when anonymizing the dicoms.
            on_error (Literal['raise', 'skip']): Whether to raise an exception when an error occurs or to skip the error.
            labels (list[str]): The labels to assign to the dicoms.
            mung_filename (Sequence[int] | Literal['all']): The parts of the filepath to keep when renaming the dicom file.
                ''all'' keeps all parts.

        Returns:
            list[str]: The list of new created dicom_ids.
        """
        files_path = APIHandler.__process_files_parameter(files_path)
        loop = asyncio.get_event_loop()
        task = self._upload_dicoms_async(files_path=files_path,
                                         batch_id=batch_id,
                                         anonymize=anonymize,
                                         anonymize_retain_codes=anonymize_retain_codes,
                                         on_error=on_error,
                                         labels=labels,
                                         mung_filename=mung_filename,
                                         )

        return loop.run_until_complete(task)

    @staticmethod
    def __process_files_parameter(file_path: str | IO | Sequence[str | IO]) -> Sequence[str | IO]:
        if isinstance(file_path, str):
            if os.path.isdir(file_path):
                file_path = [f'{file_path}/{f}' for f in os.listdir(file_path)]
        # Check if is an IO object
        elif hasattr(file_path, 'read'):
            file_path = [file_path]
        elif not hasattr(file_path, '__len__'):
            if hasattr(file_path, '__iter__'):
                file_path = list(file_path)
            else:
                file_path = [file_path]

        return file_path

    # ? maybe it is better to separate "complex" workflows to a separate class?

    def create_batch_with_dicoms(self,
                                 description: str,
                                 files_path: str | IO | Sequence[str | IO],
                                 on_error: Literal['raise', 'skip'] = 'raise',
                                 labels: Sequence[str] = None,
                                 modality: Optional[str] = None,
                                 anonymize: bool = False,
                                 anonymize_retain_codes: Sequence[tuple] = [],
                                 mung_filename: Sequence[int] | Literal['all'] = None,
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
        results = loop.run_until_complete(self._upload_dicoms_async(files_path, batch_id,
                                                                    anonymize=anonymize,
                                                                    anonymize_retain_codes=anonymize_retain_codes,
                                                                    on_error=on_error,
                                                                    labels=labels,
                                                                    mung_filename=mung_filename)
                                          )
        return batch_id, results

    def get_batch_info(self, batch_id: str) -> dict:
        """
        Get information of a batch.

        Args:
            batch_id (str): The batch unique id.

        Returns:
            dict: Informations the batch and its images.

        """
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/upload-batches/{batch_id}'
        }
        return self._run_request(request_params).json()
