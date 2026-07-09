import httpx
import pytest

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.annotationsets_api import AnnotationWorklistApi


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
        annotationsets_api = AnnotationWorklistApi(api_config, client=client)

        annotation_set_id = annotationsets_api.create(
            name="Lung Worklist",
            resource_ids=[api_ids.resource_id],
            description="Simple worklist",
            annotators=[api_ids.email],
            return_entity=False,
        )
        annotationsets_api.update_segmentation_group(
            api_ids.annotation_set_id,
            definitions,
            renames=["old_tumor:new_tumor"],
        )

    assert annotation_set_id == api_ids.annotation_set_id


def test_annotationsets_api_deprecated_parameter_aliases(
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
        if request.method == "PUT" and path == f"/annotationsets/{api_ids.annotation_set_id}/segmentation-group":
            return httpx.Response(200, json={"updated": True})
        if request.method == "GET" and path == (
            f"/annotationsets/{api_ids.annotation_set_id}/resources/{api_ids.resource_id}/segmentations"
        ):
            return httpx.Response(200, json=[])
        if request.method == "POST" and path == f"/annotationsets/{api_ids.annotation_set_id}/resources":
            return httpx.Response(200, json={"updated": True})
        if request.method == "GET" and path == f"/annotationsets/{api_ids.annotation_set_id}/users/{api_ids.email}":
            return httpx.Response(200, json={"status": "active"})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        annotationsets_api = AnnotationWorklistApi(api_config, client=client)

        with pytest.warns(DeprecationWarning, match="project_id"):
            annotationsets_api.create(
                name="Lung Worklist",
                resource_ids=[api_ids.resource_id],
                project_id=api_ids.project_id,
                return_entity=False,
            )
        with pytest.warns(DeprecationWarning, match="annotation_worklist"):
            annotationsets_api.update_segmentation_group(
                annotation_worklist=api_ids.annotation_set_id,
                definitions=definitions,
            )
        with pytest.warns(DeprecationWarning, match="annotation_set"):
            with pytest.warns(DeprecationWarning, match="resource_id"):
                with pytest.warns(DeprecationWarning, match="annotator"):
                    annotationsets_api.get_segmentations(
                        annotation_set=api_ids.annotation_set_id,
                        resource_id=api_ids.resource_id,
                        annotator=api_ids.email,
                    )
        with pytest.warns(DeprecationWarning, match="resources_to_add"):
            annotationsets_api.update_resources(
                api_ids.annotation_set_id,
                resources_to_add=[api_ids.resource_id],
            )
        with pytest.warns(DeprecationWarning, match="email"):
            annotationsets_api.get_annotator_status(
                api_ids.annotation_set_id, email=api_ids.email,
            )

    assert json_body(requests[0])["project_id"] == api_ids.project_id
    assert requests[2].url.params["annotator"] == api_ids.email
    assert json_body(requests[3])["resource_ids_to_add"] == [api_ids.resource_id]