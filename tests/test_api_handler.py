import pytest
from unittest.mock import patch
from datamint.api.client import Api
from datamint.exceptions import DatamintException
import datamint
import respx
from aioresponses import aioresponses, CallbackResult
import pydicom
from pydicom.data import get_testdata_files
import datamint.configs
from medimgkit.dicom_utils import to_bytesio
import json
from aiohttp import FormData
from typing import IO
import os
import numpy as np
from copy import deepcopy
import httpx

# pytest tests --log-cli-level=INFO

_TEST_URL = 'https://test_url.com'


def _get_request_data(request_data: dict | FormData) -> tuple[IO, str]:
    """
    Extracts the dicom bytes and the data from the request data.
    """
    if isinstance(request_data, FormData):
        for field in request_data._fields:
            if hasattr(field[-1], 'read'):
                dicom_bytes = field[-1]
                break
        data = str(request_data._fields)
    elif isinstance(request_data, dict):
        if 'dicom' in request_data:
            dicom_bytes = request_data['dicom']
        else:
            dicom_bytes = request_data['resource']
        data = str(request_data)
    else:  # Multipart object
        raise NotImplementedError
    return dicom_bytes, data


MP4_TEST_FILE = 'tests/data/test1.mp4'


class TestAPIHandler:
    @pytest.fixture
    def sample_dicom1(self) -> pydicom.Dataset:
        filepath = get_testdata_files("MR_small.dcm")[0]
        return pydicom.dcmread(filepath)

    @pytest.fixture
    def get_channels_sample(self) -> dict:
        return {'data': [{'channel_name': None,
                          'resource_data': [{'created_by': 'datamint-dev@mail.com',
                                             'customer_id': '79113ed1-0535-4f53-9359-7fe3fa9f28a8',
                                             'resource_id': '78a0bce0-9182-49d9-a780-14f444d404eb',
                                             'resource_file_name': '_%2Fdata%2Ftest_dicom2.dcm',
                                             'resource_mimetype': 'application/dicom'},
                                            {'created_by': 'datamint-dev@mail.com',
                                             'customer_id': '79113ed1-0535-4f53-9359-7fe3fa9f28a8',
                                             'resource_id': '6daa205a-5312-4104-abbb-7ec00607a53a',
                                             'resource_file_name': '_%2F_%2F_%2F_%2FVideos%2Fssstwitter.com_1709750263599.mp4',
                                             'resource_mimetype': 'video/mp4'}],
                          'deleted': False,
                          'created_at': '2024-06-03T13:28:08.584Z',
                          'updated_at': '2024-06-03T14:31:18.785Z',
                          'resource_count': '2'},
                         {'channel_name': 'test_channel',
                          'resource_data': [{'created_by': 'datamint-dev@mail.com',
                                             'customer_id': '79113ed1-0535-4f53-9359-7fe3fa9f28a8',
                                             'resource_id': 'a05fe46d-2f66-46fc-b7ef-666464ad3a28',
                                             'resource_file_name': '_%2Fdocs%2Fimages%2Flogo.png',
                                             'resource_mimetype': 'image/png'}],
                          'deleted': False,
                          'created_at': '2024-06-04T12:38:12.976Z',
                          'updated_at': '2024-06-04T12:38:12.976Z',
                          'resource_count': '1'}],
                'totalCount': '1'}

    @pytest.fixture
    def get_projects_sample(self) -> dict:
        return {
            'data': [{'id': '1c212cb4-5160-4d02-9ce9-b3dc624461f9',
                      'name': 'TestProject',
                      'description': 'Description of the project',
                      'created_at': '2025-01-28T16:55:35.379Z',
                      'created_by': 'datamint-dev@mail.com',
                      'dataset_id': 'e338b469-fa80-40cd-beb2-1e1b945a89d1',
                      'worklist_id': 'ab63f112-acf1-415d-9b54-c0f0a512553d',
                      'ai_model_id': None,
                      'viewable_ai_segs': None,
                      'editable_ai_segs': None,
                      'resource_count': '36',
                      'annotated_resource_count': '3',
                      'most_recent_experiment': None,
                      'closed_resources_count': '3',
                      'resources_to_annotate_count': 33}
                     ]
        }

    @pytest.fixture
    def get_resources_sample(self) -> dict:
        res = {'id': 'cd69c126-02ee-44af-8672-13d61b09eee4',
               'resource_uri': '/resources/cd69c126-02ee-44af-8672-13d61b09eee4/file',
               'storage': 'ImageResource',
               'location': 'resources/79113ed1-0535-4f53-9359-7fe3fa9f28a8/cd69c126-02ee-44af-8672-13d61b09eee4',
               'upload_channel': 'Unknown',
               'filename': 'normal (95).png',
               'modality': 'US',
               'mimetype': 'image/png',
               'size': 444584, 'upload_mechanism': 'api',
               'customer_id': '79113ed1-0535-4f53-9359-7fe3fa9f28a8',
               'status': 'inbox',
               'created_at': '2025-12-24T11:34:18.036Z',
               'created_by': 'datamint-dev@mail.com',
               'published': False,
               'published_on': None,
               'published_by': None,
               'publish_transforms': None,
               'deleted': False,
               'deleted_at': None,
               'deleted_by': None,
               'metadata': {'width': 693, 'height': 582},
               'source_filepath': '/tmp/normal/normal (95).png',
               'tags': ['ultrasound', 'split:train'],
               'user_info': {'firstname': None, 'lastname': None},
               'projects': [], 'labels': []}

        return {'data': [{'resources': [res]}], 'totalCount': 1}

    @pytest.fixture
    def sample_2dmask1(self) -> np.ndarray:
        """
        Creates a 32x32 mask with random values
        """
        # Use fixed seed to ensure reproducibility
        rng = np.random.RandomState(42)
        return rng.randint(0, 2, (32, 32), dtype=np.uint8)

    @respx.mock
    @patch('os.getenv')
    def test_api_handler_init(self, mock_getenv, get_projects_sample: dict):
        api_handler = Api(check_connection=False)

        ### Test wrong url ###
        with pytest.raises(DatamintException):
            api_handler = Api(server_url='wrong', check_connection=True)

    @respx.mock
    def test_get_resources(self, get_resources_sample: dict):
        from datamint.api.endpoints.resources_api import ResourcesApi

        # Mocking the response from the server
        respx.get(f"{_TEST_URL}/{ResourcesApi._ENDPOINT_BASE}").mock(
            return_value=httpx.Response(200, json=get_resources_sample)
        )

        api = Api(_TEST_URL, 'test_api_key', check_connection=False)
        resources = api.resources.get_list()
        assert len(resources) == 1
        r = resources[0]
        assert r.filename == 'normal (95).png'
