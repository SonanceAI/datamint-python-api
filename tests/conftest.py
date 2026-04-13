import json
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import unquote

import httpx
import pytest

from datamint.api.base_api import ApiConfig


TEST_URL = "https://test-url.com"


@dataclass(frozen=True)
class ApiTestIds:
    project_id: str = "11111111-1111-1111-1111-111111111111"
    dataset_id: str = "22222222-2222-2222-2222-222222222222"
    resource_id: str = "33333333-3333-3333-3333-333333333333"
    resource_id_2: str = "44444444-4444-4444-4444-444444444444"
    annotation_id: str = "55555555-5555-5555-5555-555555555555"
    annotation_id_2: str = "66666666-6666-6666-6666-666666666666"
    user_id: str = "77777777-7777-7777-7777-777777777777"
    annotation_set_id: str = "88888888-8888-8888-8888-888888888888"
    email: str = "annotator@example.com"


@pytest.fixture
def api_ids() -> ApiTestIds:
    return ApiTestIds()


@pytest.fixture
def api_config() -> ApiConfig:
    return ApiConfig(server_url=TEST_URL, api_key="test-api-key")


@pytest.fixture
def make_client() -> Callable[[Callable[[httpx.Request], httpx.Response]], httpx.Client]:
    def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
        return httpx.Client(base_url=TEST_URL, transport=httpx.MockTransport(handler))

    return _make_client


@pytest.fixture
def decoded_path() -> Callable[[httpx.Request], str]:
    def _decoded_path(request: httpx.Request) -> str:
        return unquote(request.url.path)

    return _decoded_path


@pytest.fixture
def json_body() -> Callable[[httpx.Request], dict | list | None]:
    def _json_body(request: httpx.Request) -> dict | list | None:
        if not request.content:
            return None
        return json.loads(request.content.decode("utf-8"))

    return _json_body


@pytest.fixture
def sample_resource(api_ids: ApiTestIds) -> dict:
    return {
        "id": api_ids.resource_id,
        "resource_uri": f"/resources/{api_ids.resource_id}/file",
        "storage": "ImageResource",
        "location": f"resources/customer/{api_ids.resource_id}",
        "upload_channel": "test-channel",
        "filename": "scan.png",
        "mimetype": "image/png",
        "size": 128,
        "customer_id": "99999999-9999-9999-9999-999999999999",
        "status": "inbox",
        "created_at": "2026-04-13T10:00:00Z",
        "created_by": "tester@datamint.io",
        "published": False,
        "deleted": False,
        "upload_mechanism": "api",
        "modality": "US",
        "source_filepath": "/tmp/scan.png",
        "tags": ["ultrasound"],
        "user_info": {"firstname": "Test", "lastname": "User"},
    }


@pytest.fixture
def sample_user(api_ids: ApiTestIds) -> dict:
    return {
        "email": api_ids.email,
        "firstname": "Annotator",
        "lastname": "User",
        "roles": ["annotator"],
        "customer_id": "99999999-9999-9999-9999-999999999999",
        "created_at": "2026-04-13T10:00:00Z",
    }