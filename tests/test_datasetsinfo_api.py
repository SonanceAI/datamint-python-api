from pathlib import Path

import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.datasetsinfo_api import DatasetsInfoApi


def test_datasets_api_get_resources_and_update_resources(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []
    resources_payload = [{"id": api_ids.resource_id, "filename": "scan.png"}]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "GET" and path == f"/datasets/{api_ids.dataset_id}/resources":
            return httpx.Response(200, json=resources_payload)
        if request.method == "POST" and path == f"/datasets/{api_ids.dataset_id}/resources":
            return httpx.Response(200, json={"updated": True})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        datasets_api = DatasetsInfoApi(api_config, client=client)

        resources = datasets_api.get_resources(api_ids.dataset_id, version="v2")
        datasets_api.update_resources(
            api_ids.dataset_id,
            resource_ids_to_add=[api_ids.resource_id],
            resource_ids_to_delete=[api_ids.resource_id_2],
            project_id=api_ids.project_id,
        )

    assert resources == resources_payload
    assert requests[0].url.params["version"] == "v2"
    assert json_body(requests[1]) == {
        "all_files_selected": False,
        "resource_ids_to_add": [api_ids.resource_id],
        "resource_ids_to_delete": [api_ids.resource_id_2],
        "project_id": api_ids.project_id,
    }


def test_datasets_api_download_streams_export_to_disk(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    archive_bytes = b"dataset-archive"
    output_path = tmp_path / "dataset-export.bin"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "GET" and path == f"/datasets/{api_ids.dataset_id}/download/nifti":
            return httpx.Response(200, json="download-token")
        if request.method == "GET" and path == "/datasets/download/download-token":
            return httpx.Response(
                200,
                content=archive_bytes,
                headers={"content-length": str(len(archive_bytes))},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        datasets_api = DatasetsInfoApi(api_config, client=client)
        datasets_api.download(api_ids.dataset_id, output_path, format="nifti", progress_bar=False)

    assert [decoded_path(request) for request in requests] == [
        f"/datasets/{api_ids.dataset_id}/download/nifti",
        "/datasets/download/download-token",
    ]
    assert output_path.read_bytes() == archive_bytes