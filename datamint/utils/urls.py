"""URL derivation helpers shared across the Datamint package."""

from __future__ import annotations

import logging
from urllib.parse import urlparse

_LOGGER = logging.getLogger(__name__)

# Ports used by the MLflow server depending on the target environment.
MLFLOW_PRODUCTION_PORT = 443
MLFLOW_DEVELOPMENT_PORT = 5000

_PRODUCTION_API_PREFIX = "api."
_PRODUCTION_MLFLOW_PREFIX = "mlflow."


def _bracket_ipv6(host: str) -> str:
    """Wrap an IPv6 host in brackets so it can be embedded in a URL authority.

    Args:
        host: Hostname as returned by :func:`urllib.parse.urlparse`, which
            strips the brackets from an IPv6 literal (e.g. ``::1``).

    Returns:
        ``host`` unchanged for IPv4/hostnames, or ``[host]`` when it contains
        a colon (indicating an IPv6 literal).
    """
    return f"[{host}]" if ":" in host else host


def derive_mlflow_url(api_url: str | None) -> str | None:
    """Derive the MLflow server URL from the Datamint API URL.

    The Datamint MLflow server lives on a dedicated host in production and on
    the same host as the API server in local/dev environments:

    - Production (``api.<domain>``) -> ``https://mlflow.<domain>:443``
    - Localhost/dev (anything else) -> ``http://<host>:5000``

    Args:
        api_url: Datamint API base URL, e.g. ``https://api.datamint.io`` or
            ``http://localhost:3001``. Trailing slashes are ignored.

    Returns:
        The full MLflow server URL including the port, or ``None`` if the API
        URL has no usable hostname.
    """
    if not api_url:
        return None

    hostname = urlparse(api_url.rstrip('/')).hostname
    if not hostname:
        _LOGGER.debug("Could not extract hostname from API URL: %s", api_url)
        return None

    if hostname.startswith(_PRODUCTION_API_PREFIX):
        domain = f"{_PRODUCTION_MLFLOW_PREFIX}{hostname[len(_PRODUCTION_API_PREFIX):]}"
        mlflow_url = f"https://{_bracket_ipv6(domain)}:{MLFLOW_PRODUCTION_PORT}"
    else:
        mlflow_url = f"http://{_bracket_ipv6(hostname)}:{MLFLOW_DEVELOPMENT_PORT}"

    _LOGGER.debug("Derived MLflow URL '%s' from API URL '%s'", mlflow_url, api_url)
    return mlflow_url