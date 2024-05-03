import pytest
from unittest.mock import patch
from datamintapi.api_handler import APIHandler
from datamintapi.api_handler import ResourceNotFoundError, DatamintException
import responses
from aioresponses import aioresponses, CallbackResult
import pydicom
from pydicom.dataset import FileMetaDataset
from datamintapi.utils.dicom_utils import CLEARED_STR, to_bytesio
import json
import re

_TEST_URL = 'https://test_url.com'


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
        with pytest.raises(Exception):
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

    def test_upload_dicoms(self, sample_dicom1):
        """
        Test the upload_dicoms method of the APIHandler class
        1. Test with anonymize=False
        2. Test with anonymize=True
        """
        def _callback1(url, **kwargs):
            assert kwargs['data']['batch_id'] == batch_id
            assert str(sample_dicom1.PatientName) in str(kwargs['data']['dicom'].read())
            return CallbackResult(status=201, payload={"id": "newdicomid"})

        def _callback2(url, **kwargs):
            assert kwargs['data']['batch_id'] == batch_id
            assert str(sample_dicom1.PatientName) not in str(kwargs['data']['dicom'].read())
            return CallbackResult(status=201, payload={"id": "newdicomid2"})

        batch_id = 'batchid'

        ### With anonymize=False ###
        with aioresponses() as mock_aioresp:
            # check that post request has data 'batch_id'
            mock_aioresp.post(
                f"{_TEST_URL}/dicoms",
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
                f"{_TEST_URL}/dicoms",
                callback=_callback2
            )

            new_dicoms_id = api_handler.upload_dicoms(batch_id=batch_id,
                                                      files_path=to_bytesio(sample_dicom1, 'sample_dicom1'),
                                                      anonymize=True)

            assert len(new_dicoms_id) == 1 and new_dicoms_id[0] == 'newdicomid2'

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
                return (500, "", "{}")
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
                f"{_TEST_URL}/dicoms",
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
