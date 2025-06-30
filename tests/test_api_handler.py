import pytest
from unittest.mock import patch
from datamint.apihandler.api_handler import APIHandler
import datamint
import responses
from aioresponses import aioresponses, CallbackResult
import pydicom
from pydicom.data import get_testdata_files
import datamint.configs
from datamint.utils.dicom_utils import to_bytesio
from datamint.apihandler.base_api_handler import DatamintException
import json
from aiohttp import FormData
from typing import IO
import requests
import os
import numpy as np
from copy import deepcopy
import logging

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
        res = {'id': 'f0fea905-6823-48c3-838a-b0c62c5a0466',
               'resource_uri': '/resources/f0fea905-6823-48c3-838a-b0c62c5a0466/file',
               'storage': 'ImageResource',
               'location': 'resources/f5a18308-ab64-4c1e-8a1f-db4bccd77f4a/f0fea905-6823-48c3-838a-b0c62c5a0466',
               'upload_channel': 'Unknown',
               'filename': 'IMG0002121.jpg',
               'modality': 'DX',
               'mimetype': 'image/jpeg',
               'size': 15662,
               'upload_mechanism': 'api',
               'customer_id': 'f5a18308-ab64-4c1e-8a1f-db4bccd77f4a',
               'status': 'published',
               'created_at': '2025-02-05T14:32:30.378Z',
               'created_by': 'datamint-dev@mail.com',
               'published': False,
               'published_on': '2025-02-05T14:47:47.942Z',
               'published_by': 'datamint-dev@mail.com',
               'publish_transforms': None,
               'deleted': False,
               'deleted_at': None,
               'deleted_by': None,
               'metadata': {},
               'source_filepath': 'images/IMG0002121.jpg',
               'tags': ['non_fractured', 'leg'],
               'segmentations': None,
               'measurements': None,
               'categories': None,
               'labels': [],
               'user_info': {'firstname': 'data', 'lastname': 'mint'},
               'projects': []}

        return {'data': [{'resources': [res]}]}

    @pytest.fixture
    def sample_2dmask1(self) -> np.ndarray:
        """
        Creates a 32x32 mask with random values
        """
        # Use fixed seed to ensure reproducibility
        rng = np.random.RandomState(42)
        return rng.randint(0, 2, (32, 32), dtype=np.uint8)

    @responses.activate
    @patch('os.getenv')
    def test_api_handler_init(self, mock_getenv, get_projects_sample: dict):
        def mock_getenv_side_effect(key):
            if key == datamint.configs.get_env_var_name(datamint.configs.APIKEY_KEY):
                return 'test_api_key'
            if key == datamint.configs.get_env_var_name(datamint.configs.APIURL_KEY):
                return _TEST_URL
            return None

        mock_getenv.side_effect = mock_getenv_side_effect
        api_handler = APIHandler(check_connection=False)
        assert api_handler.api_key == 'test_api_key'
        assert api_handler.root_url == _TEST_URL

        responses.get(
            f"{_TEST_URL}/projects",
            status=200,
            json=get_projects_sample
        )

        api_handler = APIHandler(check_connection=True)

        ### Test wrong url ###
        with pytest.raises(DatamintException):
            api_handler = APIHandler('wrong', check_connection=True)

    def test_upload_dicoms_resources(self, sample_dicom1):
        def _callback1(url, data, **kwargs):
            dicom_bytes, data = _get_request_data(data)
            ds = pydicom.dcmread(dicom_bytes)

            assert str(sample_dicom1.PatientName) == str(ds.PatientName)
            return CallbackResult(status=201, payload={"id": "newdicomid"})

        with aioresponses() as mock_aioresp:
            # check that post request has data 'batch_id'
            mock_aioresp.post(
                f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                callback=_callback1,
                repeat=2
            )

            api_handler = APIHandler(_TEST_URL, 'test_api_key', check_connection=False)
            new_dicoms_id = api_handler.upload_resources(files_path=to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                         channel='mychannel',
                                                         anonymize=False)
            assert new_dicoms_id == 'newdicomid'

            ### Same thing, but in a list ###
            new_dicoms_id = api_handler.upload_resources(files_path=[to_bytesio(sample_dicom1, 'sample_dicom1')],
                                                         channel='mychannel',
                                                         anonymize=False)
            assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'
            ######

    def test_upload_dicoms_mungfilename(self, sample_dicom1):
        from builtins import open

        def my_open_mock(file, *args, **kwargs):
            if file.endswith('.dcm'):
                return to_bytesio(sample_dicom1, file)
            return open(file, *args, **kwargs)

        def _request_callback(url, data, **kwargs):
            dicom_bytes, data = _get_request_data(data)
            assert '__data_test_dicom.dcm' in data
            return CallbackResult(status=201, payload={"id": "newdicomid"})

        def _request_callback2(url, data, **kwargs):
            if isinstance(data, FormData):
                data = str(data._fields)
            else:
                data = str(data)
            assert 'test_dicom.dcm' in data and 'data_test' not in data
            return CallbackResult(status=201, payload={"id": "newdicomid"})

        def _request_callback3(url, data, **kwargs):
            dicom_bytes, data = _get_request_data(data)
            assert 'me_data_test_dicom.dcm' in data
            return CallbackResult(status=201, payload={"id": "newdicomid"})

        api_handler = APIHandler(_TEST_URL, 'test_api_key', check_connection=False)

        with patch('builtins.open', new=my_open_mock):
            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback
                )
                new_dicoms_id = api_handler.upload_resources(files_path=os.path.join('..', 'data', 'test_dicom.dcm'),
                                                             anonymize=False,
                                                             mung_filename='all')
                assert new_dicoms_id == 'newdicomid'

            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback2
                )
                new_dicoms_id = api_handler.upload_resources(files_path=os.path.join('data', 'test_dicom.dcm'),
                                                             anonymize=False,
                                                             mung_filename=None)
                assert new_dicoms_id == 'newdicomid'

            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback3
                )
                new_dicoms_id = api_handler.upload_resources(files_path=os.path.join('home', 'me', 'data', 'test_dicom.dcm'),
                                                             anonymize=False,
                                                             mung_filename=[2, 3])
                assert new_dicoms_id == 'newdicomid'

    @responses.activate
    def test_upload_resources_video(self):
        def _callback(url, data, *args, **kwargs):
            assert kwargs['headers']['apikey'] == 'test_api_key'

            return CallbackResult(status=201, payload={"id": "new_resource_id"})

        with aioresponses() as mock_aioresp:
            api_handler = APIHandler(_TEST_URL, 'test_api_key', check_connection=False)
            mock_aioresp.post(
                api_handler._get_endpoint_url(APIHandler.ENDPOINT_RESOURCES),
                callback=_callback
            )

            new_resources_id = api_handler.upload_resources(files_path=MP4_TEST_FILE,
                                                            tags=['label1', 'label2'],
                                                            channel='mychannel',
                                                            anonymize=False)
            new_resources_id == 'new_resource_id'

    def test_upload_resources_assembling_dicoms(self, sample_dicom1: pydicom.Dataset):
        sample_dicom2 = deepcopy(sample_dicom1)
        sample_dicom2.InstanceNumber = 2
        sample_dicom2.SOPInstanceUID = pydicom.uid.generate_uid()
        api_handler = APIHandler(_TEST_URL, 'test_api_key', check_connection=False)

        with aioresponses() as mock_aioresp:
            mock_aioresp.post(
                f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                status=201,
                payload={"id": "new_res_id"},
                repeat=2
            )

            new_res_ids = api_handler.upload_resources(files_path=[to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                                   to_bytesio(sample_dicom2, 'sample_dicom2'),
                                                                   MP4_TEST_FILE,
                                                                   ],
                                                       channel='mychannel',
                                                       assemble_dicoms=True,
                                                       anonymize=False)
            assert len(new_res_ids) == 2

        ### Same thing, but assemble_dicoms=False ###
        with aioresponses() as mock_aioresp:
            mock_aioresp.post(
                f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                status=201,
                payload={"id": "new_res_id"},
                repeat=3
            )

            new_res_ids = api_handler.upload_resources(files_path=[to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                                   to_bytesio(sample_dicom2, 'sample_dicom2'),
                                                                   MP4_TEST_FILE,
                                                                   ],
                                                       channel='mychannel',
                                                       assemble_dicoms=False,
                                                       anonymize=False)
            assert len(new_res_ids) == 3

    @responses.activate
    def test_get_channels(self, get_channels_sample: dict):
        def _request_callback(request):
            return (200, "", json.dumps(get_channels_sample))

        # Mocking the response from the server
        responses.add_callback(
            responses.GET,
            f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}/channels",
            content_type='application/json',
            callback=_request_callback
        )

        api_handler = APIHandler(_TEST_URL, 'test_api_key', check_connection=False)
        channels_info = list(api_handler.get_channels())
        assert len(channels_info) == 2  # two channels

    @responses.activate
    def test_get_resources(self, get_resources_sample: dict):
        def _request_callback(request):
            return (200, "", json.dumps(get_resources_sample))

        # Mocking the response from the server
        responses.add_callback(
            responses.GET,
            f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
            content_type='application/json',
            callback=_request_callback
        )

        api_handler = APIHandler(_TEST_URL, 'test_api_key', check_connection=False)
        resources = list(api_handler.get_resources())
        assert len(resources) == 1
        r = resources[0]
        assert r['id'] == 'f0fea905-6823-48c3-838a-b0c62c5a0466'
