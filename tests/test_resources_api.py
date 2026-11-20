import httpx

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


def test_resources_api_filter_dicom_reports_skips_non_dicoms(
    api_config: ApiConfig,
    make_client,
    monkeypatch,
    tmp_path,
) -> None:
    jpg_path = tmp_path / "image.jpg"
    jpg_path.write_bytes(b"fake-jpg")

    def fail_if_called(_file_path):
        raise AssertionError("is_dicom_report should not be called on non-DICOM files")

    monkeypatch.setattr("datamint.api.endpoints.resources_api.is_dicom_report", fail_if_called)

    with make_client(lambda _: httpx.Response(200)) as client:
        resources_api = ResourcesApi(api_config, client=client)
        filtered, metadata = resources_api._filter_dicom_reports([str(jpg_path)], None)

    assert filtered == [str(jpg_path)]
    assert metadata is None


def test_resources_api_upload_resources_preserves_order_and_keepalive(
    api_config: ApiConfig,
    make_client,
    monkeypatch,
    tmp_path,
) -> None:
    import aiohttp

    files = []
    for i in range(3):
        p = tmp_path / f"img_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xe0" + bytes([i]))
        files.append(str(p))

    class _FakeResp:
        def __init__(self, payload: dict) -> None:
            self._payload = payload
            self.status = 200
            self.ok = True

        async def json(self):
            return self._payload

        def release(self) -> None:
            return None

    async def fake_request(self, method: str, url: str, **kwargs):
        form = kwargs["data"]
        filename = next(
            opts["filename"] for opts, _, _ in form._fields if opts.get("name") == "resource"
        )
        if "img_1" in filename:
            raise RuntimeError("upload failed")
        return _FakeResp({"id": f"id-{filename}"})

    monkeypatch.setattr(aiohttp.ClientSession, "request", fake_request)

    with make_client(lambda _: httpx.Response(200)) as client:
        resources_api = ResourcesApi(api_config, client=client)
        try:
            results = resources_api.upload_resources(
                files,
                assemble_dicoms=False,
                discard_dicom_reports=False,
                on_error="skip",
            )
            assert resources_api._aiohttp_connector is not None
            assert resources_api._aiohttp_connector._force_close is False
        finally:
            resources_api.close()

    assert results[0] == "id-img_0.jpg"
    assert isinstance(results[1], RuntimeError)
    assert results[2] == "id-img_2.jpg"