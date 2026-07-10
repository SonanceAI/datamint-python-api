import httpx
import pytest

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.resources_api import ResourcesApi


def test_resources_api_bulk_publish_and_get_not_annotated(
    api_config: ApiConfig,
    api_ids,
    sample_resource: dict,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "POST" and path == "/resources/bulk-publish":
            return httpx.Response(200, json={"published": 2})
        if request.method == "GET" and path == "/resources/not-annotated":
            return httpx.Response(
                200,
                json={"data": [{"resources": [sample_resource]}], "totalCount": 1},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        resources_api = ResourcesApi(api_config, client=client)

        resources_api.bulk_publish([api_ids.resource_id, api_ids.resource_id_2])
        resources = resources_api.get_not_annotated(limit=1, status="inbox")

    assert json_body(requests[0]) == {
        "resource_ids": [api_ids.resource_id, api_ids.resource_id_2],
    }
    assert requests[1].url.params["status"] == "inbox"
    assert requests[1].url.params["offset"] == "0"
    assert requests[1].url.params["limit"] == "1"
    assert len(resources) == 1
    assert resources[0].id == sample_resource["id"]


def test_resources_api_upload_resources_ai_model_deprecated_alias(
    api_config: ApiConfig,
    api_ids,
    make_client,
) -> None:
    with make_client(lambda request: httpx.Response(404)) as client:
        resources_api = ResourcesApi(api_config, client=client)

        # 'on_error' is intentionally invalid so the call short-circuits with a
        # ValueError right after the deprecation warning fires, without needing
        # to mock the full async upload pipeline.
        with pytest.warns(DeprecationWarning, match="ai_model"):
            with pytest.raises(ValueError):
                resources_api.upload_resources(
                    files_path=["a.dcm", "b.dcm"],
                    on_error="invalid",
                    ai_model="legacy-model",
                )