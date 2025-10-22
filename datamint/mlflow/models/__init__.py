import logging
import json
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import os
from tempfile import TemporaryDirectory
from torch import nn

_LOGGER = logging.getLogger(__name__)


def download_model_metadata(model_uri: str) -> dict:
    from mlflow.tracking.artifact_utils import get_artifact_repository

    art_repo = get_artifact_repository(artifact_uri=model_uri)
    try:
        out_artifact_path = art_repo.download_artifacts(artifact_path='metadata.json')
    except OSError as e:
        _LOGGER.warning(f"Error downloading model metadata: {e}")
        return {}

    with open(out_artifact_path, 'r') as f:
        metadata = json.load(f)
    return metadata


def _get_MLFlowLogger(trainer: L.Trainer) -> MLFlowLogger:
    for logger in trainer.loggers:
        if isinstance(logger, MLFlowLogger):
            return logger
    raise ValueError("No MLFlowLogger found in the trainer loggers.")


def log_model_metadata(metadata: dict,
                       mlflow_model: mlflow.models.Model | None = None,
                       logger: MLFlowLogger | L.Trainer | None = None,
                       model_path: str | None = None,
                       run_id: str | None = None,
                       ) -> None:
    """
    Log additional metadata to the MLflow model.
    It should be provided the one of the following combination of parameters:
    1. `mlflow_model`
    2. `logger` and `model_path`
    3. `run_id` and `model_path`

    Args:
        self: The instance of the class where this method is called.
        metadata (dict): The metadata to log.
        mlflow_model (mlflow.models.Model, optional): The MLflow model object. Defaults to None.
        logger (MLFlowLogger or L.Trainer, optional): The MLFlow logger or Lightning Trainer instance. Defaults to None.
        model_path (str, optional): The path where the model is stored in MLflow. Defaults to None.
        run_id (str, optional): The run ID of the MLflow run. Defaults to None.
    """

    # Validate inputs
    if mlflow_model is None and (logger is None or model_path is None) and (run_id is None or model_path is None):
        raise ValueError(
            "You must provide either `mlflow_model`, or both `logger` and `model_path`, "
            "or both `run_id` and `model_path`."
        )
    # not both
    if mlflow_model is not None and logger is not None:
        raise ValueError("Only one of mlflow_model or logger can be provided.")

    if logger is not None and isinstance(logger, L.Trainer):
        logger = _get_MLFlowLogger(logger)
        if logger is None:
            raise ValueError("MLFlowLogger not found in the Trainer's loggers.")
        run_id = logger.run_id
        artifact_path = model_path
        mlfclient = logger.experiment
    elif mlflow_model is not None:
        run_id = mlflow_model.run_id
        artifact_path = mlflow_model.artifact_path
        mlfclient = mlflow.client.MlflowClient()
    elif run_id is not None and model_path is not None:
        mlfclient = mlflow.client.MlflowClient()
        artifact_path = model_path
    else:
        raise ValueError("Invalid logger or mlflow_model provided.")

    with TemporaryDirectory() as tmpdir:
        metadata_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        mlfclient.log_artifact(
            run_id=run_id,
            local_path=metadata_path,
            artifact_path=artifact_path,
        )
        _LOGGER.debug(f"Additional metadata logged to {artifact_path}/metadata.json")