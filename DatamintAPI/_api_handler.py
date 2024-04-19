from typing import Optional
import os
from requests import Session
import logging
import asyncio
import aiohttp
import nest_asyncio  # For running asyncio in jupyter notebooks


_LOGGER = logging.getLogger(__name__)


class APIHandler:
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

    async def _upload_dicom_async(self, batch_id: str, file_path: str, session=None) -> str:
        # upload the file to POST /dicoms with a body that includes the batch_id
        with open(file_path, 'rb') as f:
            request_params = {
                'method': 'POST',
                'url': f'{self.root_url}/dicoms',
                # 'files': {'dicom': f},
                'data': {'batch_id': batch_id, 'dicom': f}
            }
            resp = await self._run_request_async(request_params, session)

        print(f'{file_path} uploaded')
        return resp['id']

    def upload_dicom(self, batch_id: str, file_path: str, session=None) -> str:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self._upload_dicom_async(batch_id, file_path, session))

    async def _upload_multiple_dicoms(self, files_path: list[str], batch_id: str):
        async with aiohttp.ClientSession() as session:
            loop = asyncio.get_event_loop()
            tasks = [self._upload_dicom_async(batch_id, f, session) for f in files_path]
            return await asyncio.gather(*tasks)

    # TODO: maybe it is better to separate "complex" workflows to a separate class.
    def create_new_batch(self,
                         description: str,
                         file_path: str | list[str],
                         label: str = None
                         ) -> tuple[str, list[str]]:
        """
        Create a new batch and upload the dicoms in the file_path to the batch.

        Args:
            description (str): The description of the batch
            file_path (str | list[str]): The path to the dicom file or a list of paths to dicom files.
            label (str, optional): The label of the batch. NOT USED YET. Defaults to None.

        Returns:
            tuple[str, list[str]]: The batch_id and the list of dicom_ids
        """

        if isinstance(file_path, str):
            if os.path.isdir(file_path):
                file_path = [f'{file_path}/{f}' for f in os.listdir(file_path)]
            else:
                file_path = [file_path]
            batch_id = self.upload_batch(description, len(file_path))
            loop = asyncio.get_event_loop()
            results = loop.run_until_complete(self._upload_multiple_dicoms(file_path, batch_id))
            return batch_id, results
