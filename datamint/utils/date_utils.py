from datetime import date


def to_iso_date_str(value: date | str | None) -> str | None:
    """Normalize a date filter to an ISO 8601 string.

    Args:
        value: A ``date``/``datetime`` or an ISO date string (``YYYY-MM-DD``).

    Returns:
        The ISO string, or ``None`` if ``value`` is ``None``.

    Raises:
        ValueError: If ``value`` is a string that is not a valid ISO date.
    """
    if value is None:
        return None
    if isinstance(value, str):
        date.fromisoformat(value)
        return value
    return value.isoformat()
