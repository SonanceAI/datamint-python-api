import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.annotations_api import AnnotationsApi


def test_annotations_api_patch_approve_and_delete_batch(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "PATCH" and path == f"/annotations/{api_ids.annotation_id}":
            return httpx.Response(200, json={})
        if request.method == "PATCH" and path == f"/annotations/{api_ids.annotation_id}/approve":
            return httpx.Response(200, json={"status": "approved"})
        if request.method == "POST" and path == "/annotations/delete-batch":
            return httpx.Response(200, json={"deleted": 2})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotations_api = AnnotationsApi(api_config, client=client)

        annotations_api.patch(
            api_ids.annotation_id,
            identifier="tumor",
            project_id=api_ids.project_id,
        )
        annotations_api.approve(api_ids.annotation_id)
        annotations_api.delete_batch([api_ids.annotation_id, api_ids.annotation_id_2])

    assert len(requests) == 3
    assert json_body(requests[0]) == {
        "identifier": "tumor",
        "project_id": api_ids.project_id,
    }
    assert requests[1].method == "PATCH"
    assert json_body(requests[2]) == {
        "ids": [api_ids.annotation_id, api_ids.annotation_id_2],
    }