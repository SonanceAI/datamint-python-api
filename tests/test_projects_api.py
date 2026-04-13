from pathlib import Path

import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.projects_api import ProjectsApi


def test_projects_api_members_and_annotation_statuses(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []
    members_payload = [
        {
            "project_id": api_ids.project_id,
            "user_id": api_ids.user_id,
            "roles": ["PROJECT_ANNOTATOR"],
        }
    ]
    statuses_payload = [
        {
            "resource_id": api_ids.resource_id,
            "status": "annotated",
            "user_id": api_ids.user_id,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/members":
            return httpx.Response(200, json=members_payload)
        if request.method == "POST" and path == f"/projects/{api_ids.project_id}/members/{api_ids.user_id}":
            return httpx.Response(200, json={"updated": True})
        if request.method == "DELETE" and path == f"/projects/{api_ids.project_id}/members/{api_ids.user_id}":
            return httpx.Response(200, json={"removed": True})
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/annotation-statuses":
            return httpx.Response(200, json=statuses_payload)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        projects_api = ProjectsApi(api_config, client=client)

        members = projects_api.get_members(api_ids.project_id)
        projects_api.set_member(api_ids.project_id, api_ids.user_id, ["PROJECT_ANNOTATOR"])
        projects_api.remove_member(api_ids.project_id, api_ids.user_id)
        statuses = projects_api.get_annotation_statuses(
            api_ids.project_id,
            status="annotated",
            user_id=api_ids.user_id,
            resource_id=api_ids.resource_id,
        )

    assert members == members_payload
    assert statuses == statuses_payload
    assert json_body(requests[1]) == {"roles": ["PROJECT_ANNOTATOR"]}
    assert requests[3].url.params["status"] == "annotated"
    assert requests[3].url.params["user_id"] == api_ids.user_id
    assert requests[3].url.params["resource_id"] == api_ids.resource_id


def test_projects_api_download_annotations_streams_export_to_disk(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []
    export_bytes = b"resource_id,status\n1,annotated\n"
    output_path = tmp_path / "annotations.csv"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/download-annotations":
            return httpx.Response(
                200,
                content=export_bytes,
                headers={"content-length": str(len(export_bytes))},
            )
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        projects_api = ProjectsApi(api_config, client=client)
        projects_api.download_annotations(
            api_ids.project_id,
            output_path,
            format="csv",
            annotators=[api_ids.email],
            annotations=["tumor"],
            from_date="2026-04-01",
            to_date="2026-04-13",
            progress_bar=False,
        )

    request = requests[0]
    assert request.url.params["format"] == "csv"
    assert request.url.params.get_list("annotators[]") == [api_ids.email]
    assert request.url.params.get_list("annotations[]") == ["tumor"]
    assert request.url.params["from"] == "2026-04-01"
    assert request.url.params["to"] == "2026-04-13"
    assert output_path.read_bytes() == export_bytes