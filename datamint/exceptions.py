class DatamintException(Exception):
    """Base class for all Datamint exceptions."""
    pass


# ---------------------------------------------------------------------------
# Auth / access
# ---------------------------------------------------------------------------

class AuthenticationError(DatamintException):
    """Raised when the API key is missing or rejected (HTTP 401)."""
    pass


class PermissionDeniedError(DatamintException):
    """Raised when the authenticated user lacks permission for the requested operation (HTTP 403)."""
    pass


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
# Input validation
# ---------------------------------------------------------------------------

class ValidationError(DatamintException):
    """Raised when the server rejects a request due to invalid input (HTTP 400/422)."""
    pass


# ---------------------------------------------------------------------------
# Network / connectivity
# ---------------------------------------------------------------------------

class NetworkError(DatamintException):
    """Raised on connection failures, SSL errors, or other transport-level problems."""
    pass


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
# Async job timeouts
# ---------------------------------------------------------------------------

class JobTimeoutError(DatamintException, TimeoutError):
    """Raised when a deployment or inference job does not finish within the allowed time.

    Subclasses both DatamintException and the built-in TimeoutError so callers
    catching either one will handle it correctly.
    """
    pass
