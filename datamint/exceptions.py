class DatamintException(Exception):
    """Base class for all Datamint exceptions."""


# ---------------------------------------------------------------------------
# Auth / access
# ---------------------------------------------------------------------------

class AuthenticationError(DatamintException):
    """Raised when the API key is missing or rejected (HTTP 401)."""


class PermissionDeniedError(DatamintException):
    """Raised when the authenticated user lacks permission for the requested operation (HTTP 403)."""


# ---------------------------------------------------------------------------
# Resource state
# ---------------------------------------------------------------------------

class ItemNotFoundError(DatamintException):
    """Raised when a requested item does not exist (HTTP 404)."""

    def __init__(self, item_type: str, params: dict):
        self.item_type = item_type
        self.params = params

    @property
    def resource_type(self):
        return self.item_type

    @resource_type.setter
    def resource_type(self, value: str):  # Alias kept for backward compatibility.
        self.item_type = value

    def set_params(self, resource_type: str, params: dict):
        self.item_type = resource_type
        self.params = params

    def __str__(self):
        return f"Item '{self.item_type}' not found for parameters: {self.params}"


ResourceNotFoundError = ItemNotFoundError  # Alias kept for backward compatibility.


class EntityAlreadyExistsError(DatamintException):
    """Raised when trying to create an entity that already exists."""

    def __init__(self, entity_type: str, params: dict):
        super().__init__()
        self.entity_type = entity_type
        self.params = params

    def __str__(self) -> str:
        return f"Entity '{self.entity_type}' already exists for parameters: {self.params}"


# ---------------------------------------------------------------------------
# Client-side session state
# ---------------------------------------------------------------------------

class DefaultProjectNotSetError(DatamintException):
    """Raised when a method requires a project, none was passed, and no default
    project has been selected via `datamint.select_project()` (or the selected
    default could not be found on this connection)."""

    def __init__(self, hint: str | None = None):
        self.hint = hint
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.hint:
            return (f"No project specified, and the default project '{self.hint}' "
                    f"(set via select_project()) could not be found on this "
                    f"connection. Pass project=... explicitly, or call "
                    f"select_project() with a valid project.")
        return ("No project specified and no default project is set. Pass "
                "project=... explicitly, or call "
                "datamint.select_project('<name-or-id>') once per session.")


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class ValidationError(DatamintException):
    """Raised when the server rejects a request due to invalid input (HTTP 400/422)."""


# ---------------------------------------------------------------------------
# Network / connectivity
# ---------------------------------------------------------------------------

class NetworkError(DatamintException):
    """Raised on connection failures, SSL errors, or other transport-level problems."""


# ---------------------------------------------------------------------------
# Server-side failures
# ---------------------------------------------------------------------------

class ServerError(DatamintException):
    """Raised when the server returns an unexpected error (HTTP 5xx)."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code

    def __str__(self) -> str:
        if self.status_code:
            return f"Server error {self.status_code}: {super().__str__()}"
        return super().__str__()


# ---------------------------------------------------------------------------
# Model deployment / inference
# ---------------------------------------------------------------------------

class ModelNotDeployedError(DatamintException):
    """Raised when trying to run inference on a model with no deployed image (HTTP 404)."""

    def __init__(
        self,
        model_name: str,
        model_version: int | None = None,
        model_alias: str | None = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.model_alias = model_alias
        super().__init__(str(self))

    def __str__(self) -> str:
        if self.model_version is not None:
            ref = f"{self.model_name}:{self.model_version}"
            deploy_kwarg = f"model_version={self.model_version}"
        elif self.model_alias is not None:
            ref = f"{self.model_name}:{self.model_alias}"
            deploy_kwarg = f"model_alias='{self.model_alias}'"
        else:
            ref = f"{self.model_name}:champion"
            deploy_kwarg = None

        deploy_call = f"api.deploy_model.start('{self.model_name}'"
        if deploy_kwarg:
            deploy_call += f", {deploy_kwarg}"
        deploy_call += ")"

        return (
            f"Model '{ref}' is not deployed, so it cannot run inference yet. "
            f"Deploy it first with {deploy_call}, wait for the job to finish "
            f"(api.deploy_model.wait(job)), then retry."
        )


# ---------------------------------------------------------------------------
# Async job timeouts
# ---------------------------------------------------------------------------

class JobTimeoutError(DatamintException, TimeoutError):
    """Raised when a deployment or inference job does not finish within the allowed time.

    Subclasses both DatamintException and the built-in TimeoutError so callers
    catching either one will handle it correctly.
    """
