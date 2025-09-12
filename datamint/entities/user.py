from .base_entity import BaseEntity

class User(BaseEntity):
    """User entity model.

    Attributes:
        email: User email address (unique identifier in most cases).
        firstname: First name.
        lastname: Last name.
        roles: List of role strings assigned to the user.
        customer_id: UUID of the owning customer/tenant.
        created_at: ISO 8601 timestamp of creation.
    """
    email: str
    firstname: str | None
    lastname: str | None
    roles: list[str]
    customer_id: str
    created_at: str

    # Potential improvement: convert created_at to datetime for easier comparisons.