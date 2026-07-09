from pathlib import Path

import httpx
import pytest

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
        projects_api.set_member(api_ids.user_id, ["PROJECT_ANNOTATOR"], project=api_ids.project_id)
        projects_api.remove_member(api_ids.user_id, project=api_ids.project_id)
        statuses = projects_api.get_annotation_statuses(
            api_ids.project_id,
            status="annotated",
            user_id=api_ids.user_id,
            resource=api_ids.resource_id,
        )

    assert members == members_payload
    assert statuses == statuses_payload
    assert json_body(requests[1]) == {"roles": ["PROJECT_ANNOTATOR"]}
    assert requests[3].url.params["status"] == "annotated"
    assert requests[3].url.params["user_id"] == api_ids.user_id
    assert requests[3].url.params["resource_id"] == api_ids.resource_id


def test_projects_api_pinned_metrics(
    api_config: ApiConfig,
    api_ids,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []
    project_payload = {
        "id": api_ids.project_id,
        "name": "Test Project",
        "created_at": "2026-04-13T10:00:00Z",
        "created_by": "tester@datamint.io",
        "dataset_id": "dataset-1",
        "archived": False,
        "resource_count": 1,
        "description": None,
        "pinned_metrics": ["val/accuracy", "val/f1"],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "PATCH" and path == f"/projects/{api_ids.project_id}":
            return httpx.Response(200, json={})
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}":
            return httpx.Response(200, json=project_payload)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        projects_api = ProjectsApi(api_config, client=client)

        projects_api.set_pinned_metrics(["val/accuracy", "val/f1"], project=api_ids.project_id)
        pinned = projects_api.get_pinned_metrics(project=api_ids.project_id)

        project = projects_api.get_by_id(api_ids.project_id)
        project.set_pinned_metrics(["val/iou", "val/dice"])

    assert json_body(requests[0]) == {"pinned_metrics": ["val/accuracy", "val/f1"]}
    assert pinned == ["val/accuracy", "val/f1"]
    assert json_body(requests[3]) == {"pinned_metrics": ["val/iou", "val/dice"]}


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
            output_path,
            format="csv",
            annotators=[api_ids.email],
            annotations=["tumor"],
            from_date="2026-04-01",
            to_date="2026-04-13",
            progress_bar=False,
            project=api_ids.project_id,
        )

    request = requests[0]
    assert request.url.params["format"] == "csv"
    assert request.url.params.get_list("annotators[]") == [api_ids.email]
    assert request.url.params.get_list("annotations[]") == ["tumor"]
    assert request.url.params["from"] == "2026-04-01"
    assert request.url.params["to"] == "2026-04-13"
    assert output_path.read_bytes() == export_bytes


def test_projects_api_deprecated_parameter_aliases(
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
        if request.method == "GET" and path == "/projects":
            return httpx.Response(200, json=[])
        if request.method == "POST" and path == "/projects":
            return httpx.Response(200, json={"id": api_ids.project_id})
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/annotation-statuses":
            return httpx.Response(200, json=[])
        if request.method == "DELETE" and path == (
            f"/projects/{api_ids.project_id}/resources/{api_ids.resource_id}"
            f"/annotator/{api_ids.email}/status-reset"
        ):
            return httpx.Response(200, json={})
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/users/{api_ids.email}/status":
            return httpx.Response(200, json={"status": "active"})
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/annotators-statistic":
            return httpx.Response(200, json=[])
        if request.method == "GET" and path == f"/projects/{api_ids.project_id}/reviewmessages":
            return httpx.Response(200, json=[])
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        projects_api = ProjectsApi(api_config, client=client)

        with pytest.warns(DeprecationWarning, match="resources_ids"):
            projects_api.create(
                name="Legacy Project",
                description="desc",
                resources_ids=[api_ids.resource_id],
                return_entity=False,
            )
        with pytest.warns(DeprecationWarning, match="resource_id"):
            projects_api.get_annotation_statuses(api_ids.project_id, resource_id=api_ids.resource_id)
        with pytest.warns(DeprecationWarning, match="annotator"):
            projects_api.reset_annotator_status(
                api_ids.resource_id, project=api_ids.project_id, annotator=api_ids.email,
            )
        with pytest.warns(DeprecationWarning, match="email"):
            projects_api.get_annotator_status(email=api_ids.email, project=api_ids.project_id)
        with pytest.warns(DeprecationWarning, match="email"):
            projects_api.get_annotators_stats(project=api_ids.project_id, email=api_ids.email)
        with pytest.warns(DeprecationWarning, match="annotator"):
            projects_api.get_review_messages(project=api_ids.project_id, annotator=api_ids.email)

    assert json_body(requests[1])["resource_ids"] == [api_ids.resource_id]
    assert requests[2].url.params["resource_id"] == api_ids.resource_id