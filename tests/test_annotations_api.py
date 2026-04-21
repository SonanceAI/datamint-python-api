import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.annotations_api import AnnotationsApi
from datamint.entities.annotations import ImageClassification, LineAnnotation


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


def test_annotations_api_create_accepts_annotation_entity(
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
        if request.method == "POST" and path == f"/annotations/{api_ids.resource_id}/annotations":
            return httpx.Response(200, json=[api_ids.annotation_id])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotations_api = AnnotationsApi(api_config, client=client)
        created_id = annotations_api.create(
            api_ids.resource_id,
            ImageClassification(
                name="category",
                value="benign",
                imported_from="integration-test",
                model_id="model-123",
            ),
        )

    assert created_id == api_ids.annotation_id
    assert json_body(requests[0]) == [
        {
            "value": "benign",
            "type": "category",
            "identifier": "category",
            "scope": "image",
            "imported_from": "integration-test",
            "is_model": True,
            "model_id": "model-123",
        }
    ]


def test_annotations_api_get_by_id_returns_typed_geometry_annotation(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
) -> None:
    payload = {
        "id": api_ids.annotation_id,
        "resource_id": api_ids.resource_id,
        "name": "Line1",
        "type": "line",
        "index": 2,
        "geometry": {"points": [[0, 0, 2], [10, 30, 2]]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        path = decoded_path(request).rstrip("/")
        if request.method == "GET" and path == f"/annotations/{api_ids.annotation_id}":
            return httpx.Response(200, json=payload)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotations_api = AnnotationsApi(api_config, client=client)
        annotation = annotations_api.get_by_id(api_ids.annotation_id)

    assert isinstance(annotation, LineAnnotation)
    assert annotation.frame_index == 2
    assert annotation.scope == "frame"
    assert annotation.geometry is not None
    assert annotation.geometry.point1 == (0, 0, 2)
    assert annotation.geometry.point2 == (10, 30, 2)