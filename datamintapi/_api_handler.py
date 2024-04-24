from typing import Optional, IO, Sequence
import os
from requests import Session
import logging
import asyncio
import aiohttp
import nest_asyncio  # For running asyncio in jupyter notebooks
from datamintapi.dicom_utils import anonymize_dicom
import pydicom
from io import BytesIO

_LOGGER = logging.getLogger(__name__)


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

    def upload_batch(self,
                     description: str,
                     size: int,
                     modality: Optional[str] = None,
                     session=None) -> str:
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

    async def _upload_dicom_async(self, batch_id: str,
                                  file_path: str | IO,
                                  anonymize: bool = False,
                                  anonymize_retain_codes: Sequence[tuple] = [],
                                  labels: list[str] = None,
                                  session=None) -> str:
        if anonymize:
            ds = pydicom.dcmread(file_path)
            ds = anonymize_dicom(ds, retain_codes=anonymize_retain_codes)
            # make the dicom `ds` object a file-like object in order to avoid unnecessary disk writes
            f = BytesIO()
            pydicom.dcmwrite(f, ds)
            f.name = file_path
            f.mode = 'rb'
            f.seek(0)
        elif isinstance(file_path, str):
            f = open(file_path, 'rb')
        else:
            f = file_path

        try:
            request_params = {
                'method': 'POST',
                'url': f'{self.root_url}/dicoms',
                'data': {'batch_id': batch_id, 'dicom': f}
            }

            if labels is not None:
                request_params['data']['labels[]'] = str(labels)
            resp = await self._run_request_async(request_params, session)

            print(f'{file_path} uploaded')
            return resp['id']
        except Exception as e:
            raise e
        finally:
            f.close()

    def upload_dicom(self, batch_id: str, file_path: str | IO, session=None) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._upload_dicom_async(batch_id, file_path, session))

    async def _upload_multiple_dicoms(self,
                                      files_path: list[str | IO],
                                      batch_id: str,
                                      anonymize: bool = False,
                                      anonymize_retain_codes: Sequence[tuple] = [],
                                      labels=None
                                      ):
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(10)  # Limit to 10 parallel requests

            async def __upload_single_dicom(file_path):
                async with semaphore:
                    return await self._upload_dicom_async(
                        batch_id, file_path, anonymize, anonymize_retain_codes,
                        labels=labels,
                        session=session,
                    )
            tasks = [__upload_single_dicom(f) for f in files_path]
            return await asyncio.gather(*tasks)

    # TODO: maybe it is better to separate "complex" workflows to a separate class.
    def create_new_batch(self,
                         description: str,
                         file_path: str | IO | Sequence[str | IO],
                         labels: Sequence[str] = None,
                         anonymize: bool = False,
                         anonymize_retain_codes: Sequence[tuple] = []
                         ) -> tuple[str, list[str]]:
        """
        Create a new batch and upload the dicoms in the file_path to the batch.

        Args:
            description (str): The description of the batch
            file_path (str | IO | Sequence[str | IO]): The path to the dicom file or a list of paths to dicom files.
            label (Sequence[str]): The label of the batch. NOT USED YET. Defaults to None.

        Returns:
            tuple[str, list[str]]: The batch_id and the list of created dicom_ids.
        """
        if labels is not None:
            labels = [l.strip() for l in labels]

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

        batch_id = self.upload_batch(description, len(file_path))
        loop = asyncio.get_event_loop()
        results = loop.run_until_complete(self._upload_multiple_dicoms(file_path, batch_id,
                                                                       anonymize=anonymize,
                                                                       anonymize_retain_codes=anonymize_retain_codes,
                                                                       labels=labels)
                                          )
        return batch_id, results

    def get_batch_info(self, batch_id: str) -> dict:
        request_params = {
            'method': 'GET',
            'url': f'{self.root_url}/upload-batches/{batch_id}'
        }
        return self._run_request(request_params).json()
