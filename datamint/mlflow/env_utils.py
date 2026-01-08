"""
Utility functions for automatically configuring MLflow environment variables
based on Datamint configuration.
"""

import os
import logging
from urllib.parse import urlparse
from datamint import configs


_LOGGER = logging.getLogger(__name__)


def get_datamint_api_url() -> str | None:
    """Get the Datamint API URL from configuration or environment variables."""
    api_url = configs.get_value(configs.APIURL_KEY, include_envvars=True)  # configs checks env vars first
    return api_url


def get_datamint_api_key() -> str | None:
    """Get the Datamint API key from configuration or environment variables."""
    # First check environment variable
    api_key = os.getenv('DATAMINT_API_KEY')
    if api_key:
        return api_key

    # Then check configuration
    api_key = configs.get_value(configs.APIKEY_KEY)
    if api_key:
        return api_key

    return None


def _get_mlflowdatamint_uri() -> str | None:
    api_url = get_datamint_api_url()
    if not api_url:
        return None
    _LOGGER.debug(f"Retrieved Datamint API URL: {api_url}")

    # Remove trailing slash if present
    api_url = api_url.rstrip('/')
    # api_url samples:
    # https://api.datamint.io
    # http://localhost:3001

    parsed_url = urlparse(api_url)
    base_url = f"{parsed_url.scheme}://{parsed_url.hostname}"
    _LOGGER.debug(f"Derived base URL for MLflow Datamint: {base_url}")
    # FIXME: It should work with https or datamint-api server should forward https requests.
    base_url = base_url.replace('https://', 'http://')
    if len(base_url.replace('http:', '')) == 0:
        return None

    mlflow_uri = f"{base_url}:5000"
    return mlflow_uri


def setup_mlflow_environment(overwrite: bool = False,
                             set_mlflow: bool = True) -> bool:
    """
    Set up MLflow environment variables based on Datamint configuration.

    Args:
        overwrite (bool): If True, overwrite existing MLflow environment variables.
        set_mlflow (bool): If True, set the MLflow tracking URI using mlflow.set_tracking_uri().

    Returns:
        bool: True if success, False otherwise.
    """
    api_key = get_datamint_api_key()
    mlflow_uri = _get_mlflowdatamint_uri()
    _LOGGER.debug(f"Setting up MLflow environment variables from Datamint configuration: URI={mlflow_uri}, API_KEY={'***' if api_key is not None else None}")
    if not mlflow_uri or not api_key:
        _LOGGER.warning("Datamint configuration incomplete, cannot auto-configure MLflow")
        return False

    if overwrite or not os.getenv('MLFLOW_TRACKING_TOKEN'):
        os.environ['MLFLOW_TRACKING_TOKEN'] = api_key
    if overwrite or not os.getenv('MLFLOW_TRACKING_URI'):
        os.environ['MLFLOW_TRACKING_URI'] = mlflow_uri

    if set_mlflow:
        import mlflow
        mlflow.set_tracking_uri(mlflow_uri)

    return True


def ensure_mlflow_configured() -> None:
    """
    Ensure MLflow environment is properly configured.
    Raises ValueError if configuration is incomplete.
    """
    if not setup_mlflow_environment():
        if not os.getenv('MLFLOW_TRACKING_URI'):
            raise ValueError(
                "MLflow environment not configured. Please either:\n"
                "1. Run 'datamint-config --default-url <url>',  or\n"
                "2. Set DATAMINT_API_URL environment variable, or\n"
                "3. Manually set MLFLOW_TRACKING_URI environment variable"
            )
        if not os.getenv('MLFLOW_TRACKING_TOKEN'):
            raise ValueError(
                "MLflow environment not configured. Please either:\n"
                "1. Run 'datamint-config', or\n"
                "2. Set DATAMINT_API_KEY environment variable, or\n"
                "3. Manually set MLFLOW_TRACKING_TOKEN environment variable"
            )
