import pytest
from unittest.mock import patch
from datamintapi._api_handler import APIHandler
import responses


class TestAPIHandler:
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
            "https://test_url.com/upload-batches",
            json={"id": "test_id"},
            status=201,
        )
        api_handler = APIHandler('https://test_url.com', 'test_api_key')
        batch_id = api_handler.upload_batch('test_description', 2)
        assert batch_id == 'test_id'
