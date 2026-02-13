# Monkey patch mlflow.tracking._tracking_service.utils.get_tracking_uri
from .tracking.fluent import set_project
import mlflow.tracking._tracking_service.utils as mlflow_utils
from functools import wraps
import logging
from .env_utils import setup_mlflow_environment, ensure_mlflow_configured
from typing import TYPE_CHECKING

_LOGGER = logging.getLogger(__name__)

# Store reference to original function
_original_get_tracking_uri = mlflow_utils.get_tracking_uri
_SETUP_CALLED_SUCCESSFULLY = False

if mlflow_utils.is_tracking_uri_set():
    _LOGGER.warning("MLflow tracking URI is already set before patching get_tracking_uri.")


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
    if mlflow_utils.is_tracking_uri_set():
        _LOGGER.warning("MLflow tracking URI is already set before patching get_tracking_uri.")
    try:
        _SETUP_CALLED_SUCCESSFULLY = setup_mlflow_environment(set_mlflow=True)
    except Exception as e:
        _SETUP_CALLED_SUCCESSFULLY = False
        _LOGGER.error("Failed to set up MLflow environment: %s", e)
    ret = _original_get_tracking_uri(*args, **kwargs)
    return ret


setup_mlflow_environment(set_mlflow=False)
# Replace the original function with our patched version
mlflow_utils.get_tracking_uri = _patched_get_tracking_uri

_ALREADY_CONFIGURED_LOGGING = False


def _configure_mlflow_loggers():
    global _ALREADY_CONFIGURED_LOGGING
    if _ALREADY_CONFIGURED_LOGGING:
        return

    from mlflow.environment_variables import MLFLOW_LOGGING_LEVEL
    from mlflow.utils.logging_utils import SuppressLogFilter
    import logging.config
    import rich.logging
    import os

    if 'MLFLOW_SCORING_SERVER_REQUEST_TIMEOUT' not in os.environ:
        # probably not running in mlflow server, so no need to configure our mlflow loggers
        return

    _ALREADY_CONFIGURED_LOGGING = True
    mlflow_log_level = (MLFLOW_LOGGING_LEVEL.get() or "INFO").upper()

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "handlers": {
                "datamint_mlflow_handler": {
                    "class": "rich.logging.RichHandler",
                    "filters": ["suppress_in_thread"],
                },
            },
            "loggers": {
                'datamint.mlflow': {
                    "handlers": ["datamint_mlflow_handler"],
                    "level": mlflow_log_level,
                    "propagate": False,
                },
            },
            "filters": {
                "suppress_in_thread": {
                    "()": SuppressLogFilter,
                }
            },
        }
    )
    _LOGGER.info("Configured MLflow loggers with RichHandler and level %s", mlflow_log_level)

try:
    _configure_mlflow_loggers()
except Exception as e:
    _LOGGER.error("Failed to configure MLflow loggers: %s", e)

if TYPE_CHECKING:
    from .flavors.model import DatamintModel
else:
    import lazy_loader as lazy

    __getattr__, __dir__, __all__ = lazy.attach(
        __name__,
        submodules=['flavors.model', 'flavors.datamint_flavor'],
        submod_attrs={
            "flavors.model": ["DatamintModel"],
            "flavors.datamint_flavor": ["log_model", "load_model"],
        },
    )


__all__ = ['set_project', 'setup_mlflow_environment', 'ensure_mlflow_configured', 'DatamintModel']
