from typing import Optional
import os
from requests import Session
import requests
import logging

_LOGGER = logging.getLogger(__name__)


class APIHandler:
    DATAMINT_API_VENV_NAME = 'DATAMINT_API_KEY'

    def __init__(self,
                 root_url: str,
                 api_key: Optional[str] = None):
        self.root_url = root_url
        self.api_key = api_key if api_key is not None else os.getenv(APIHandler.DATAMINT_API_VENV_NAME)
        if self.api_key is None:
            _LOGGER.warning("API key not provided!")

    def _run_request(self,
                     request_args: dict,
                     session=None) -> requests.Response:
        if session is None:
            with Session() as session:
                return self._run_request(request_args, session)

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
                     modality: str,
                     session) -> str:
        post_params = {'description': description,
                       'size': size,
                       'modality': modality
                       }
        request_params = {
            'method': 'POST',
            'url': f'{self.root_url}/upload-batches',
            'json': post_params
        }

        resp = self._run_request(session, request_params)
        return resp.json()['id']

    def upload_dicom(self, batch_id: str, file_path: str, session) -> str:
        # TODO: make it async
        # upload the file to POST /dicoms with a body that includes the batch_id
        with open(file_path, 'rb') as f:
            request_params = {
                'method': 'POST',
                'url': f'{self.root_url}/dicoms',
                'files': {'dicom': f},
                'data': {'batch_id': batch_id}
            }
            resp = self._run_request(session, request_params)
        return resp.json()['id']
