"""
Utility functions for automatically configuring MLflow environment variables
based on Datamint configuration.
"""

import os
import logging
from typing import Optional
from datamint import configs


_LOGGER = logging.getLogger(__name__)


def get_datamint_api_url() -> Optional[str]:
    """Get the Datamint API URL from configuration or environment variables."""
    # First check environment variable
    api_url = os.getenv('DATAMINT_API_URL')
    if api_url:
        return api_url

    # Then check configuration
    api_url = configs.get_value(configs.APIURL_KEY)
    if api_url:
        return api_url

    return None


def get_datamint_api_key() -> Optional[str]:
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


def _get_mlflowdatamint_uri() -> Optional[str]:
    api_url = get_datamint_api_url()
    if not api_url:
        return None

    # Remove trailing slash if present
    api_url = api_url.rstrip('/')

    base_url = api_url.rsplit(':', maxsplit=1)[0]
    base_url = base_url.replace('https://', 'http://')  # FIXME

    mlflow_uri = f"{base_url}:5000"
    return mlflow_uri


def setup_mlflow_environment(overwrite: bool = False,
                             set_mlflow: bool = True) -> bool:
    """
    Automatically set up MLflow environment variables based on Datamint configuration.

    Returns:
        bool: True if MLflow environment was successfully configured, False otherwise.
    """
    _LOGGER.debug("Setting up MLflow environment variables from Datamint configuration")
    api_key = get_datamint_api_key()
    mlflow_uri = _get_mlflowdatamint_uri()
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
    Raises an exception if configuration is incomplete.
    """
    if not setup_mlflow_environment():
        if not os.getenv('MLFLOW_TRACKING_URI') or not os.getenv('MLFLOW_TRACKING_TOKEN'):
            raise ValueError(
                "MLflow environment not configured. Please either:\n"
                "1. Run 'datamint-config' to set up Datamint configuration, or\n"
                "2. Set DATAMINT_API_URL and DATAMINT_API_KEY environment variables, or\n"
                "3. Manually set MLFLOW_TRACKING_URI and MLFLOW_TRACKING_TOKEN environment variables"
            )
