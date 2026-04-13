import httpx

from datamint.api.base_api import ApiConfig
from datamint.api.endpoints.users_api import UsersApi


def test_users_api_invite_get_and_manage_invitations(
    api_config: ApiConfig,
    api_ids,
    sample_user: dict,
    make_client,
    decoded_path,
    json_body,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = decoded_path(request)
        if request.method == "POST" and path == "/users/invite":
            return httpx.Response(200, json={"status": "sent"})
        if request.method == "GET" and path == f"/users/{api_ids.email}":
            return httpx.Response(200, json=sample_user)
        if request.method == "GET" and path == "/users/invitations":
            return httpx.Response(200, json=[{"email": api_ids.email, "project_id": api_ids.project_id}])
        if request.method == "DELETE" and path == f"/users/invitations/{api_ids.email}/revoke":
            return httpx.Response(200, json={"revoked": True})
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with make_client(handler) as client:
        users_api = UsersApi(api_config, client=client)

        invite_result = users_api.invite(
            api_ids.email,
            firstname="Annotator",
            lastname="User",
            return_url="https://app.datamint.io/return",
            project_id=api_ids.project_id,
            project_roles=["PROJECT_ANNOTATOR"],
            annotation_worklist_id=api_ids.annotation_set_id,
        )
        user = users_api.get_by_email(api_ids.email)
        invitations = users_api.get_invitations(project_id=api_ids.project_id)
        users_api.revoke_invitation(api_ids.email)

    assert invite_result == {"status": "sent"}
    assert user.email == sample_user["email"]
    assert user.roles == sample_user["roles"]
    assert invitations == [{"email": api_ids.email, "project_id": api_ids.project_id}]

    assert len(requests) == 4
    assert requests[0].method == "POST"
    assert json_body(requests[0]) == {
        "email": api_ids.email,
        "firstname": "Annotator",
        "lastname": "User",
        "return_url": "https://app.datamint.io/return",
        "project_id": api_ids.project_id,
        "project_roles": ["PROJECT_ANNOTATOR"],
        "annotation_worklist_id": api_ids.annotation_set_id,
    }
    assert requests[2].url.params["project_id"] == api_ids.project_id
    assert requests[3].method == "DELETE"