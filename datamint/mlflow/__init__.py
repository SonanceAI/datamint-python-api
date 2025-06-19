from .tracking.fluent import set_project
from .env_utils import setup_mlflow_environment, ensure_mlflow_configured

# Automatically set up MLflow environment when the module is imported
setup_mlflow_environment()

__all__ = ['set_project', 'setup_mlflow_environment', 'ensure_mlflow_configured']


