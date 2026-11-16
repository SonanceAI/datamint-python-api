from datetime import date
from unittest.mock import MagicMock

import aiohttp
import httpx
import numpy as np
import pytest

from datamint.api.base_api import ApiConfig
from datamint.api.dto import CreateAnnotationDto
from datamint.api.endpoints.annotations_api import AnnotationsApi
from datamint.entities import Resource
from datamint.entities.annotations import (
    CategoryAnnotation,
    ImageClassification,
    ImageSegmentation,
    LineAnnotation,
    VolumeSegmentation,
)
from tests.conftest import TEST_URL


class _FakeAiohttpResponse:
    def __init__(self, json_payload, status: int = 200) -> None:
        self._json_payload = json_payload
        self.status = status
        self.ok = 200 <= status < 400

    async def json(self):
        return self._json_payload

    def raise_for_status(self) -> None:
        return None

    def release(self) -> None:
        return None


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
            project=api_ids.project_id,
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


def test_annotations_api_get_list_date_range(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and decoded_path(request) == "/annotations/search":
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotations_api = AnnotationsApi(api_config, client=client)

        annotations_api.get_list(from_date=date(2026, 1, 1), to_date=date(2026, 1, 31))

    assert json_body(requests[0])["from"] == "2026-01-01"
    assert json_body(requests[0])["to"] == "2026-01-31"


def test_create_annotation_dto_model_uuid() -> None:
    raw_uuid = "AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE"
    canonical_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    dto = CreateAnnotationDto(
        type="category",
        identifier="finding",
        scope="image",
        value="positive",
        model_uuid=raw_uuid,
    )
    payload = dto.to_dict()
    assert payload["model_uuid"] == canonical_uuid
    assert payload["is_model"] is True
    assert "model_id" not in payload

    dto_without_model = CreateAnnotationDto(
        type="category",
        identifier="finding",
        scope="image",
        value="positive",
    )
    assert "model_uuid" not in dto_without_model.to_dict()

    with pytest.raises(ValueError):
        CreateAnnotationDto(
            type="category",
            identifier="finding",
            scope="image",
            model_uuid="not-a-uuid",
        )

    with pytest.raises(ValueError):
        CreateAnnotationDto(
            type="category",
            identifier="finding",
            scope="image",
            is_model=False,
            model_uuid=canonical_uuid,
        )


def test_annotations_api_json_endpoints_serialize_model_uuid(
    api_config: ApiConfig,
    api_ids,
    sample_resource: dict,
    make_client,
    decoded_path,
    json_body,
) -> None:
    model_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    resource = Resource(**sample_resource)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "POST" and path == f"/annotations/{api_ids.resource_id}/annotations":
            return httpx.Response(200, json=[api_ids.annotation_id])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotations_api = AnnotationsApi(api_config, client=client)

        annotations_api.create(
            api_ids.resource_id,
            CategoryAnnotation(
                name="category",
                value="benign",
                model_id="Baseline",
                model_uuid=model_uuid,
            ),
        )
        annotations_api.create_image_classification(
            api_ids.resource_id,
            identifier="finding",
            value="normal",
            model_id="Baseline",
            model_uuid=model_uuid,
            source="model_deploy",
        )
        annotations_api.add_category_annotation(
            api_ids.resource_id,
            identifier="grade",
            value="high",
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.add_text_annotation(
            api_ids.resource_id,
            identifier="notes",
            value="clear",
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.add_label_annotation(
            api_ids.resource_id,
            identifier="fracture",
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.create_numeric_annotation(
            api_ids.resource_id,
            identifier="score",
            value=0.95,
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.add_box_annotation(
            (0, 0),
            (10, 20),
            resource=resource,
            identifier="lesion",
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.add_line_annotation(
            (0, 0),
            (10, 20),
            resource=resource,
            identifier="diameter",
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.add_point_annotation(
            (5, 10),
            resource=resource,
            identifier="center",
            model_id="Baseline",
            model_uuid=model_uuid,
        )
        annotations_api.add_polyline_annotation(
            [(0, 0), (10, 0), (10, 10)],
            resource=resource,
            identifier="contour",
            closed=True,
            model_id="Baseline",
            model_uuid=model_uuid,
        )

    assert len(requests) == 10
    for req in requests:
        body = json_body(req)
        assert isinstance(body, list)
        assert body[0]["is_model"] is True
        assert body[0]["model_id"] == "Baseline"
        assert body[0]["model_uuid"] == model_uuid


def test_upload_segmentations_with_model_uuid_skips_check_model(
    api_config: ApiConfig,
    api_ids,
    make_client,
    monkeypatch,
) -> None:
    model_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    models_api = MagicMock()
    models_api.get_by_name.side_effect = AssertionError("ModelsApi.get_by_name must not be called")

    mask = np.ones((4, 4), dtype=np.uint8)
    create_url = f"{TEST_URL}/annotations/{api_ids.resource_id}/annotations"
    file_url = f"{TEST_URL}/annotations/{api_ids.resource_id}/annotations/{api_ids.annotation_id}/file"
    aiohttp_calls: list[tuple[str, str, dict]] = []

    async def fake_request(self, method: str, url: str, **kwargs):
        aiohttp_calls.append((method, url, kwargs))
        if url == create_url:
            return _FakeAiohttpResponse([api_ids.annotation_id])
        if url == file_url:
            return _FakeAiohttpResponse({"ok": True})
        raise AssertionError(f"Unexpected aiohttp request: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "request", fake_request)

    with make_client(lambda _: httpx.Response(500)) as client:
        annotations_api = AnnotationsApi(api_config, client=client, models_api=models_api)
        try:
            ids = annotations_api.upload_segmentations(
                resource=api_ids.resource_id,
                file_path=mask,
                name="liver",
                model_name="Baseline",
                source="model_deploy",
                model_uuid=model_uuid.upper(),
            )
        finally:
            annotations_api.close()

    assert ids == [api_ids.annotation_id]
    models_api.get_by_name.assert_not_called()

    create_calls = [kwargs for method, url, kwargs in aiohttp_calls if method == "POST" and url == create_url]
    assert len(create_calls) == 1
    sent_json = create_calls[0]["json"]
    assert sent_json[0]["model_id"] == "Baseline"
    assert sent_json[0]["model_uuid"] == model_uuid
    assert sent_json[0]["is_model"] is True
    assert sent_json[0]["source"] == "model_deploy"


def test_upload_volume_segmentation_with_model_uuid_skips_check_model(
    api_config: ApiConfig,
    api_ids,
    make_client,
    monkeypatch,
) -> None:
    model_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    models_api = MagicMock()
    models_api.get_by_name.side_effect = AssertionError("ModelsApi.get_by_name must not be called")

    volume = np.ones((4, 4, 2), dtype=np.uint8)
    upload_url = f"{TEST_URL}/annotations/{api_ids.resource_id}/segmentations/file"
    aiohttp_calls: list[tuple[str, str, dict]] = []

    async def fake_request(self, method: str, url: str, **kwargs):
        aiohttp_calls.append((method, url, kwargs))
        if url == upload_url:
            return _FakeAiohttpResponse([api_ids.annotation_id])
        raise AssertionError(f"Unexpected aiohttp request: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "request", fake_request)

    with make_client(lambda _: httpx.Response(500)) as client:
        annotations_api = AnnotationsApi(api_config, client=client, models_api=models_api)
        try:
            ids = annotations_api.upload_volume_segmentation(
                resource=api_ids.resource_id,
                file_path=volume,
                name={1: "liver"},
                model_name="Baseline",
                source="model_deploy",
                model_uuid=model_uuid,
            )
        finally:
            annotations_api.close()

    assert ids == [api_ids.annotation_id]
    models_api.get_by_name.assert_not_called()

    upload_calls = [kwargs for method, url, kwargs in aiohttp_calls if method == "POST" and url == upload_url]
    assert len(upload_calls) == 1
    form = upload_calls[0]["data"]
    assert isinstance(form, aiohttp.FormData)
    fields = {opts["name"]: value for opts, _, value in form._fields}
    assert fields["model_id"] == "Baseline"
    assert fields["model_uuid"] == model_uuid
    assert fields["source"] == "model_deploy"


def test_upload_predictions_forwards_model_uuid_across_types(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
    monkeypatch,
) -> None:
    model_uuid = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    models_api = MagicMock()
    models_api.get_by_name.side_effect = AssertionError("ModelsApi.get_by_name must not be called")

    httpx_requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        httpx_requests.append(request)
        if request.method == "POST" and decoded_path(request) == f"/annotations/{api_ids.resource_id}/annotations":
            return httpx.Response(200, json=[api_ids.annotation_id_2])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    vol_seg = VolumeSegmentation(
        name="vol",
        segmentation_data=np.ones((4, 4, 2), dtype=np.uint8),
        class_map={1: "liver"},
    )
    img_seg = ImageSegmentation(
        name="lesion",
        segmentation_data=np.ones((4, 4), dtype=np.uint8),
        frame_index=0,
    )
    cat_ann = CategoryAnnotation(name="finding", value="positive")

    vol_url = f"{TEST_URL}/annotations/{api_ids.resource_id}/segmentations/file"
    create_url = f"{TEST_URL}/annotations/{api_ids.resource_id}/annotations"
    file_url = f"{TEST_URL}/annotations/{api_ids.resource_id}/annotations/{api_ids.annotation_id}/file"
    aiohttp_calls: list[tuple[str, str, dict]] = []

    async def fake_request(self, method: str, url: str, **kwargs):
        aiohttp_calls.append((method, url, kwargs))
        if url in (vol_url, create_url):
            return _FakeAiohttpResponse([api_ids.annotation_id])
        if url == file_url:
            return _FakeAiohttpResponse({"ok": True})
        raise AssertionError(f"Unexpected aiohttp request: {method} {url}")

    monkeypatch.setattr(aiohttp.ClientSession, "request", fake_request)

    with make_client(handler) as client:
        annotations_api = AnnotationsApi(api_config, client=client, models_api=models_api)
        try:
            ids = annotations_api.upload_predictions(
                resource=api_ids.resource_id,
                predictions=[vol_seg, img_seg, cat_ann],
                model_name="Baseline",
                source="model_deploy",
                model_uuid=model_uuid,
            )
        finally:
            annotations_api.close()

    assert ids == [api_ids.annotation_id, api_ids.annotation_id, api_ids.annotation_id_2]
    models_api.get_by_name.assert_not_called()

    # VolumeSegmentation multipart form
    vol_calls = [kwargs for method, url, kwargs in aiohttp_calls if method == "POST" and url == vol_url]
    assert len(vol_calls) == 1
    vol_fields = {opts["name"]: value for opts, _, value in vol_calls[0]["data"]._fields}
    assert vol_fields["model_id"] == "Baseline"
    assert vol_fields["model_uuid"] == model_uuid

    # ImageSegmentation JSON create
    seg_calls = [kwargs for method, url, kwargs in aiohttp_calls if method == "POST" and url == create_url]
    assert len(seg_calls) == 1
    seg_json = seg_calls[0]["json"]
    assert seg_json[0]["model_id"] == "Baseline"
    assert seg_json[0]["model_uuid"] == model_uuid

    # CategoryAnnotation JSON create
    assert len(httpx_requests) == 1
    cat_json = json_body(httpx_requests[0])
    assert cat_json[0]["model_id"] == "Baseline"
    assert cat_json[0]["model_uuid"] == model_uuid
    assert cat_json[0]["is_model"] is True
    assert cat_json[0]["source"] == "model_deploy"
