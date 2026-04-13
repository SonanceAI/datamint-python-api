from ..entity_base_api import CreatableEntityApi, ApiConfig
from datamint.entities import User
import httpx


class UsersApi(CreatableEntityApi[User]):
    def __init__(self,
                 config: ApiConfig,
                 client: httpx.Client | None = None) -> None:
        super().__init__(config, User, 'users', client)

    def create(self,
               email: str,
               password: str | None = None,
               firstname: str | None = None,
               lastname: str | None = None,
               roles: list[str] | None = None,
               exists_ok: bool = False
               ) -> str:
        """Create a new user.

        Args:
            email: The user's email address.
            password: The user's password. If None, a random password will be generated.
            firstname: The user's first name.
            lastname: The user's last name.
            roles: List of roles to assign to the user.
            exists_ok: If ``True``, do not raise an error when a user with the same
                email already exists. Instead, the existing user's id is returned when
                possible.

        Returns:
            The id of the created user.
        """
        data = dict(
            email=email,
            password=password,
            firstname=firstname,
            lastname=lastname,
            roles=roles
        )
        return self._create(data, exists_ok=exists_ok)

    def invite(self,
               email: str,
               firstname: str | None = None,
               lastname: str | None = None,
               return_url: str | None = None,
               project_id: str | None = None,
               project_roles: list[str] | None = None,
               annotation_worklist_id: str | None = None,
               ) -> dict:
        """Send an invitation email to a new user.

        Args:
            email: The invitee's email address.
            firstname: The invitee's first name.
            lastname: The invitee's last name.
            return_url: URL the invite link should redirect to after acceptance.
            project_id: Optional project to add the invitee to.
            project_roles: Roles to assign in the given project.
            annotation_worklist_id: Optional annotation worklist to associate.

        Returns:
            The server response as a dict.
        """
        payload: dict = {'email': email}
        if firstname is not None:
            payload['firstname'] = firstname
        if lastname is not None:
            payload['lastname'] = lastname
        if return_url is not None:
            payload['return_url'] = return_url
        if project_id is not None:
            payload['project_id'] = project_id
        if project_roles is not None:
            payload['project_roles'] = project_roles
        if annotation_worklist_id is not None:
            payload['annotation_worklist_id'] = annotation_worklist_id
        response = self._make_request('POST', f'/{self.endpoint_base}/invite', json=payload)
        return response.json()

    def get_by_email(self, email: str) -> User:
        """Get a user by email address.

        Args:
            email: The user's email address.

        Returns:
            The User instance.
        """
        response = self._make_request('GET', f'/{self.endpoint_base}/{email}')
        return self._init_entity_obj(**response.json())

    def update_user(self, email: str, **kwargs) -> None:
        """Partially update a user's profile.

        Args:
            email: The user's email address.
            **kwargs: Fields to update (e.g. ``firstname``, ``lastname``,
                ``roles``).
        """
        payload = {k: v for k, v in kwargs.items() if v is not None}
        self._make_request('PATCH', f'/{self.endpoint_base}/{email}', json=payload)

    def delete_user(self, email: str) -> None:
        """Delete a user by email address.

        Args:
            email: The user's email address.
        """
        self._make_request('DELETE', f'/{self.endpoint_base}/{email}')

    def get_invitations(self, project_id: str | None = None) -> list[dict]:
        """List pending user invitations.

        Args:
            project_id: Optional project ID to filter invitations.

        Returns:
            List of invitation dicts.
        """
        params: dict = {}
        if project_id is not None:
            params['project_id'] = project_id
        response = self._make_request('GET', f'/{self.endpoint_base}/invitations',
                                      params=params or None)
        return response.json()

    def revoke_invitation(self, email: str) -> None:
        """Revoke a pending invitation by the invitee's email.

        Args:
            email: The invitee's email address.
        """
        self._make_request('DELETE', f'/{self.endpoint_base}/invitations/{email}/revoke')
