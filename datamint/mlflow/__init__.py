from .tracking.fluent import set_project
from .env_utils import setup_mlflow_environment, ensure_mlflow_configured

# Monkey patch mlflow.tracking._tracking_service.utils.get_tracking_uri
import mlflow.tracking._tracking_service.utils as mlflow_utils
from functools import wraps
import logging

_LOGGER = logging.getLogger(__name__)

# Store reference to original function
_original_get_tracking_uri = mlflow_utils.get_tracking_uri
_SETUP_CALLED_SUCCESSFULLY = False


@wraps(_original_get_tracking_uri)
def _patched_get_tracking_uri(*args, **kwargs):
    """Patched version of get_tracking_uri that ensures MLflow environment is set up first.

    This wrapper ensures that setup_mlflow_environment is called before any tracking
    URI operations, guaranteeing proper MLflow configuration.

    Args:
        *args: Arguments passed to the original get_tracking_uri function.
        **kwargs: Keyword arguments passed to the original get_tracking_uri function.

    Returns:
        The result of the original get_tracking_uri function.
    """
    global _SETUP_CALLED_SUCCESSFULLY
    if _SETUP_CALLED_SUCCESSFULLY:
        return _original_get_tracking_uri(*args, **kwargs)
    try:
        _SETUP_CALLED_SUCCESSFULLY = setup_mlflow_environment()
    except Exception as e:
        _SETUP_CALLED_SUCCESSFULLY = False
        _LOGGER.error("Failed to set up MLflow environment: %s", e)
    ret = _original_get_tracking_uri(*args, **kwargs)
    return ret


# Replace the original function with our patched version
mlflow_utils.get_tracking_uri = _patched_get_tracking_uri

__all__ = ['set_project', 'setup_mlflow_environment', 'ensure_mlflow_configured']
