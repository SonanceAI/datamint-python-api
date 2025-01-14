import pytest
from unittest.mock import patch, mock_open
from datamintapi.apihandler.api_handler import APIHandler
import responses
from aioresponses import aioresponses, CallbackResult
import pydicom
from pydicom.dataset import FileMetaDataset
from datamintapi.utils.dicom_utils import to_bytesio
import json
from aiohttp import FormData
from typing import Tuple
from io import BytesIO
import requests
import os

_TEST_URL = 'https://test_url.com'


def _get_request_data(request_data) -> Tuple[BytesIO, str]:
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
    def sample_dicom1(self):
        ds = pydicom.Dataset()
        ds.preamble = b"\0" * 128
        ds.PatientName = "John Doe"
        ds.PatientID = "12345"
        ds.PatientWeight = 70
        ds.Modality = "CT"
        ds.SOPInstanceUID = pydicom.uid.generate_uid()
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.file_meta = FileMetaDataset()
        ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.file_meta.MediaStorageSOPClassUID = "1"
        ds.file_meta.ImplementationClassUID = "1.1"
        ds.file_meta.ImplementationVersionName = "1.1"
        ds.file_meta.FileMetaInformationGroupLength = 196
        ds.file_meta.FileMetaInformationVersion = b'\x00\x01'

        return ds

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

    @patch('os.getenv')
    def test_api_handler_init(self, mock_getenv):
        mock_getenv.return_value = 'test_api_key'
        api_handler = APIHandler('test_url')
        assert api_handler.api_key == 'test_api_key'

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
                callback=_callback1
            )

            api_handler = APIHandler(_TEST_URL, 'test_api_key')
            new_dicoms_id = api_handler.upload_resources(files_path=to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                         channel='mychannel',
                                                         anonymize=False)
            assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

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

        api_handler = APIHandler(_TEST_URL, 'test_api_key')

        with patch('builtins.open', new=my_open_mock):
            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback
                )
                new_dicoms_id = api_handler.upload_resources(files_path=os.path.join('..', 'data', 'test_dicom.dcm'),
                                                             anonymize=False,
                                                             mung_filename='all')
                assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback2
                )
                new_dicoms_id = api_handler.upload_resources(files_path=os.path.join('data', 'test_dicom.dcm'),
                                                             anonymize=False,
                                                             mung_filename=None)
                assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback3
                )
                new_dicoms_id = api_handler.upload_resources(files_path=os.path.join('home', 'me', 'data', 'test_dicom.dcm'),
                                                             anonymize=False,
                                                             mung_filename=[2, 3])
                assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

    @responses.activate
    def test_get_batches(self):
        api_handler = APIHandler(_TEST_URL, 'test_api_key')

        batches_data = [
            {"id": "batch1", "description": "Batch 1"},
            {"id": "batch2", "description": "Batch 2"},
            {"id": "batch3", "description": "Batch 3"}
        ]
        responses.get(
            f"{_TEST_URL}/upload-batches",
            json={"data": batches_data},
            status=200,
        )

        batches = list(api_handler.get_batches())
        assert len(batches) == 3

    @responses.activate
    def test_get_batches_no_batch(self):
        api_handler = APIHandler(_TEST_URL, 'test_api_key')

        responses.get(
            f"{_TEST_URL}/upload-batches",
            json={"data": []},
            status=200,
        )

        batches = list(api_handler.get_batches())
        assert len(batches) == 0

    @responses.activate
    def test_upload_segmentation(self, sample_dicom1):
        from builtins import open

        def my_open_mock(file, *args, **kwargs):
            if file == 'test_segmentation_file.nifti':
                return to_bytesio(sample_dicom1, file)
            return open(file, *args, **kwargs)
        # Mocking the response from the server
        responses.post(
            f"{_TEST_URL}/segmentations",
            json={"id": "test_segmentation_id"},
            status=201,
        )
        api_handler = APIHandler(_TEST_URL, 'test_api_key')
        dicom_id = 'test_dicom_id'
        segmentation_name = 'test_segmentation_name'
        file_path = 'test_segmentation_file.nifti'
        with patch('builtins.open', new=my_open_mock):
            segmentation_id = api_handler.upload_segmentation(dicom_id, file_path, segmentation_name)
        assert segmentation_id == 'test_segmentation_id'

    @responses.activate
    def test_upload_resources_video(self):
        def _callback(url, data, *args, **kwargs):
            assert kwargs['headers']['apikey'] == 'test_api_key'

            return CallbackResult(status=201, payload={"id": "new_resource_id"})

        with aioresponses() as mock_aioresp:
            # check that post request has data 'batch_id'
            api_handler = APIHandler(_TEST_URL, 'test_api_key')
            mock_aioresp.post(
                api_handler._get_endpoint_url(APIHandler.ENDPOINT_RESOURCES),
                callback=_callback
            )

            new_resources_id = api_handler.upload_resources(files_path=MP4_TEST_FILE,
                                                            labels=['label1', 'label2'],
                                                            channel='mychannel',
                                                            anonymize=False)
            assert len(new_resources_id) == 1 and new_resources_id[0] == 'new_resource_id'

    @responses.activate
    def test_wrong_url(self, get_channels_sample: dict):
        def _request_callback(request):
            if 'wrong_url' in request.url:
                return (404, "", "404 Client Error: Not Found for url:")
            return (200, "", json.dumps(get_channels_sample))

        # Mocking the response from the server
        responses.add_callback(
            responses.GET,
            f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}/channels",
            content_type='application/json',
            callback=_request_callback
        )
        responses.add_callback(
            responses.GET,
            f"https://wrong_url/{APIHandler.ENDPOINT_RESOURCES}/channels",
            content_type='application/json',
            callback=_request_callback
        )

        api_handler = APIHandler(_TEST_URL, 'test_api_key')
        list(api_handler.get_channels())

        with pytest.raises(requests.exceptions.HTTPError):
            api_handler = APIHandler('https://wrong_url', 'test_api_key')
            list(api_handler.get_channels())

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

        api_handler = APIHandler(_TEST_URL, 'test_api_key')
        channels_info = list(api_handler.get_channels())
        assert len(channels_info) == 2  # two channels
