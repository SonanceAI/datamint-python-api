import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.annotationsets_api import AnnotationSetsApi


def test_annotationsets_api_create_update_and_update_segmentation_group(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []
    definitions = [{"identifier": "tumor", "color": [255, 0, 0], "index": 1}]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "POST" and path == "/annotationsets":
            return httpx.Response(200, json={"id": api_ids.annotation_set_id})
        if request.method == "PATCH" and path == f"/annotationsets/{api_ids.annotation_set_id}":
            return httpx.Response(200, json={"updated": True})
        if request.method == "PUT" and path == f"/annotationsets/{api_ids.annotation_set_id}/segmentation-group":
            return httpx.Response(200, json={"updated": True})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotationsets_api = AnnotationSetsApi(api_config, client=client)

        annotation_set_id = annotationsets_api.create(
            name="Lung Worklist",
            resource_ids=[api_ids.resource_id],
            description="Simple worklist",
            annotators=[api_ids.email],
        )
        annotationsets_api.update(
            api_ids.annotation_set_id,
            name="Updated Worklist",
            status="active",
        )
        annotationsets_api.update_segmentation_group(
            api_ids.annotation_set_id,
            definitions,
            renames=["old_tumor:new_tumor"],
        )

    assert annotation_set_id == api_ids.annotation_set_id
    assert json_body(requests[0]) == {
        "name": "Lung Worklist",
        "resource_ids": [api_ids.resource_id],
        "description": "Simple worklist",
        "annotators": [api_ids.email],
    }
    assert json_body(requests[1]) == {"name": "Updated Worklist", "status": "active"}
    assert json_body(requests[2]) == {
        "segmentationData": {
            "segmentationValueType": "single_label",
            "definitions": definitions,
        },
        "renames": ["old_tumor:new_tumor"],
    }