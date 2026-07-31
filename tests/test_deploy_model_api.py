import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.deploy_model_api import DeployModelApi


def test_deploy_model_api_stream_status(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
) -> None:
    requests: list[httpx.Request] = []
    sse_body = b'data: {"status": "running"}\n\n'

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        expected = f"/datamint/api/v1/deploy-model/status/{api_ids.resource_id}/stream"
        if request.method == "GET" and path == expected:
            return httpx.Response(200, content=sse_body)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        deploy_api = DeployModelApi(api_config, client=client)

        events = list(deploy_api.stream_status(job=api_ids.resource_id))

    assert events == [{"status": "running"}]
    assert len(requests) == 1
