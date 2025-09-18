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
               roles: list[str] | None = None
               ) -> str:
        """Create a new user.

        Args:
            email: The user's email address.
            password: The user's password. If None, a random password will be generated.
            firstname: The user's first name.
            lastname: The user's last name.
            roles: List of roles to assign to the user.

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
        return self._create(data)
