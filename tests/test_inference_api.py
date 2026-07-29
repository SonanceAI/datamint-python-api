import httpx
import pytest

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.inference_api import InferenceApi
from datamint.exceptions import ItemNotFoundError, ModelNotDeployedError


def test_predict_image_raises_model_not_deployed_error(
    api_config: ApiConfig,
    make_client,
    decoded_path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = decoded_path(request)
        if request.method == "POST" and path == "/datamint/api/v1/model-inference/predict-image":
            return httpx.Response(
                404,
                json={"detail": "No deployed image found for model 'my_model:champion'. Please deploy the model first."},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        inference_api = InferenceApi(api_config, client=client)

        with pytest.raises(ModelNotDeployedError) as excinfo:
            inference_api.predict_image("my_model", resource_id="11111111-1111-1111-1111-111111111111")

    err = excinfo.value
    assert err.model_name == "my_model"
    assert err.model_version is None
    assert err.model_alias is None
    assert "not deployed" in str(err)
    assert "api.deploy_model.start('my_model')" in str(err)


def test_predict_image_unrelated_404_is_not_translated(
    api_config: ApiConfig,
    make_client,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"detail": "Model 'my_model' not found"})

    with make_client(handler) as client:
        inference_api = InferenceApi(api_config, client=client)

        with pytest.raises(ItemNotFoundError):
            inference_api.predict_image("my_model", resource_id="11111111-1111-1111-1111-111111111111")


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
