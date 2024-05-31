import pytest
from unittest.mock import patch, mock_open
from datamintapi.api_handler import APIHandler
from datamintapi.api_handler import ResourceNotFoundError, DatamintException
import responses
from aioresponses import aioresponses, CallbackResult
import pydicom
from pydicom.dataset import FileMetaDataset
from datamintapi.utils.dicom_utils import to_bytesio
import json
import re
from aiohttp import FormData
from typing import Tuple
from io import BytesIO

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

    @patch('os.getenv')
    def test_api_handler_init(self, mock_getenv):
        mock_getenv.return_value = 'test_api_key'
        api_handler = APIHandler('test_url')
        assert api_handler.api_key == 'test_api_key'

        ### Second test case: Expecting an exception to be raised, since no api key is provided ###
        mock_getenv.return_value = None
        with pytest.raises(DatamintException):
            APIHandler('test_url')

    @responses.activate
    def test_upload_batch(self):
        # Mocking the response from the server
        responses.post(
            f"{_TEST_URL}/upload-batches",
            json={"id": "test_id"},
            status=201,
        )
        api_handler = APIHandler(_TEST_URL, 'test_api_key')
        batch_id = api_handler.create_batch('test_description', 2)
        assert batch_id == 'test_id'

    def test_upload_dicoms_batch_id(self, sample_dicom1):
        """
        Test the upload_dicoms method of the APIHandler class
        1. Test with anonymize=False
        2. Test with anonymize=True
        """
        def _callback1(url, data, **kwargs):
            dicom_bytes, data = _get_request_data(data)
            ds = pydicom.dcmread(dicom_bytes)

            assert batch_id in data
            assert str(sample_dicom1.PatientName) == str(ds.PatientName)
            return CallbackResult(status=201, payload={"id": "newdicomid"})

        def _callback2(url, data, **kwargs):
            dicom_bytes, data = _get_request_data(data)
            ds = pydicom.dcmread(dicom_bytes)

            assert batch_id in data
            assert str(sample_dicom1.PatientName) not in str(ds.PatientName)
            return CallbackResult(status=201, payload={"id": "newdicomid2"})

        batch_id = 'batchid'

        ### With anonymize=False ###
        with aioresponses() as mock_aioresp:
            # check that post request has data 'batch_id'
            mock_aioresp.post(
                f"{_TEST_URL}/{APIHandler.ENDPOINT_DICOMS}",
                callback=_callback1
            )

            api_handler = APIHandler(_TEST_URL, 'test_api_key')
            new_dicoms_id = api_handler.upload_dicoms(batch_id=batch_id,
                                                      files_path=to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                      anonymize=False)
            assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

        ### With anonymize=True ###
        with aioresponses() as mock_aioresp:
            # check that post request has data 'batch_id'
            mock_aioresp.post(
                f"{_TEST_URL}/{APIHandler.ENDPOINT_DICOMS}",
                callback=_callback2
            )

            new_dicoms_id = api_handler.upload_dicoms(batch_id=batch_id,
                                                      files_path=to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                      anonymize=True)

            assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid2'

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
            new_dicoms_id = api_handler.upload_dicoms(files_path=to_bytesio(sample_dicom1, 'sample_dicom1'),
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
            assert 'data/test_dicom.dcm' in data
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
                new_dicoms_id = api_handler.upload_dicoms(files_path='../data/test_dicom.dcm',
                                                          anonymize=False,
                                                          mung_filename='all')
                assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback2
                )
                new_dicoms_id = api_handler.upload_dicoms(files_path='data/test_dicom.dcm',
                                                          anonymize=False,
                                                          mung_filename=None)
                assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

            with aioresponses() as mock_aioresp:
                mock_aioresp.post(
                    f"{_TEST_URL}/{APIHandler.ENDPOINT_RESOURCES}",
                    callback=_request_callback3
                )
                new_dicoms_id = api_handler.upload_dicoms(files_path='/home/me/data/test_dicom.dcm',
                                                          anonymize=False,
                                                          mung_filename=[2, 3])
                assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid'

    @responses.activate
    def test_create_batch_with_dicoms(self, sample_dicom1):
        uploaded_batches_mock = []  # used to store the batch_id of the uploaded batches

        def _request_callback(request):
            uploaded_batches_mock.append('batch_id')
            resp_body = {"id": 'batch_id'}
            return (201, "", json.dumps(resp_body))

        def _request_callback_batchinfo(request):
            batch_id = request.url.split('/')[-1]
            if batch_id not in uploaded_batches_mock:
                return (404, "", "{}")
            resp = {"id": batch_id,
                    "description": "description",
                    "images": [{"filename": "sample_dicom1"}]
                    }
            return (200, "", json.dumps(resp))

        # Mocking the response from the server
        responses.add_callback(
            responses.POST,
            f"{_TEST_URL}/upload-batches",
            content_type='application/json',
            callback=_request_callback
        )

        responses.add_callback(
            responses.GET,
            re.compile(f"{_TEST_URL}/upload-batches/.+"),
            content_type='application/json',
            callback=_request_callback_batchinfo

        )

        with aioresponses() as mock_aioresp:
            mock_aioresp.post(
                f"{_TEST_URL}/{APIHandler.ENDPOINT_DICOMS}",
                payload={"id": "newdicomid"},
                status=201
            )

            api_handler = APIHandler(_TEST_URL, 'test_api_key')
            dc1_io = to_bytesio(sample_dicom1, 'sample_dicom1')
            batch_id, dicoms_ids = api_handler.create_batch_with_dicoms('description',
                                                                        files_path=[dc1_io],
                                                                        anonymize=False
                                                                        )

            assert batch_id == 'batch_id'
            assert len(dicoms_ids) == 1 and dicoms_ids[0] == 'newdicomid'

        batchinfo = api_handler.get_batch_info(batch_id)
        assert batchinfo['description'] == 'description'
        assert len(batchinfo['images']) == 1
        assert batchinfo['images'][0]['filename'] == 'sample_dicom1'

        # Test non existing batch
        with pytest.raises(ResourceNotFoundError):
            api_handler.get_batch_info('non_existing_batch_id')

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

