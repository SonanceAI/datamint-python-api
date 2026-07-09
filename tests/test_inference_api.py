import httpx
import pytest

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.inference_api import InferenceApi


def test_inference_api_get_status_and_stream_status_job_id_deprecated_alias(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
) -> None:
    requests: list[httpx.Request] = []
    sse_body = b'data: {"status": "completed"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        base = f"/datamint/api/v1/model-inference/status/{api_ids.resource_id}"
        if request.method == "GET" and path == base:
            return httpx.Response(
                200,
                json={"job_id": api_ids.resource_id, "status": "completed", "model_name": "my-model"},
            )
        if request.method == "GET" and path == f"{base}/stream":
            return httpx.Response(200, content=sse_body)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        inference_api = InferenceApi(api_config, client=client)

        with pytest.warns(DeprecationWarning, match="job_id"):
            job = inference_api.get_status(job_id=api_ids.resource_id)
        with pytest.warns(DeprecationWarning, match="job_id"):
            events = list(inference_api.stream_status(job_id=api_ids.resource_id))

    assert job.id == api_ids.resource_id
    assert events == [{"status": "completed"}]
