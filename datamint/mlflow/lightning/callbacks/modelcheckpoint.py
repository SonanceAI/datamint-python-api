from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from weakref import proxy
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from typing import Literal, Any
import inspect
import torch
from torch import nn
import lightning.pytorch as L
from datamint.mlflow.models import log_model_metadata, _get_MLFlowLogger
from datamint.mlflow.env_utils import ensure_mlflow_configured
import mlflow
import logging
from lightning.pytorch.loggers import MLFlowLogger

_LOGGER = logging.getLogger(__name__)


def help_infer_signature(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in x.items()}
    elif isinstance(x, list):
        return [v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in x]
    elif isinstance(x, tuple):
        return tuple(v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for v in x)

    return x


class MLFlowModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args,
                 register_model_name: str | None = None,
                 register_model_on: Literal["train", "val", "test", "predict"] | None = None,
                 code_paths: list[str] | None = None,
                 log_model_at_end_only: bool = True,
                 additional_metadata: dict[str, Any] | None = None,
                 extra_pip_requirements: list[str] | None = None,
                 **kwargs):
        """
        MLFlowModelCheckpoint is a custom callback for PyTorch Lightning that integrates with MLFlow to log and register models.

        Args:
            register_model_name (str | None): The name to register the model under in MLFlow. If None, the model will not be registered.
            register_model_on (Literal["train", "val", "test", "predict"] | None): The stage at which to register the model. If None, the model will not be registered.
            code_paths (list[str] | None): List of paths to Python files that should be included in the MLFlow model.
            log_model_at_end_only (bool): If True, only log the model to MLFlow at the end of the training instead of after every checkpoint save.
            additional_metadata (dict[str, Any] | None): Additional metadata to log with the model as a JSON file.
            extra_pip_requirements (list[str] | None): Additional pip requirements to include with the MLFlow model.
            **kwargs: Keyword arguments for ModelCheckpoint.
        """
        # Ensure MLflow is configured when callback is initialized
        ensure_mlflow_configured()

        super().__init__(*args, **kwargs)
        if self.save_top_k > 1:
            raise NotImplementedError("save_top_k > 1 is not supported. "
                                      "Please use save_top_k=1 to save only the best model.")
        if self.save_last is not None and self.save_top_k != 0 and self.monitor is not None:
            raise NotImplementedError("save_last is not supported with monitor and save_top_k!=0. "
                                      "Please use two separate callbacks: one for saving the last model "
                                      "and another for saving the best model based on the monitor metric.")

        if register_model_name is not None and register_model_on is None:
            raise ValueError("If you provide a register_model_name, you must also provide a register_model_on.")
        if register_model_on is not None and register_model_name is None:
            raise ValueError("If you provide a register_model_on, you must also provide a register_model_name.")
        if register_model_on not in ["train", "val", "test", "predict", None]:
            raise ValueError("register_model_on must be one of train, val, test or predict.")

        self.register_model_name = register_model_name
        self.register_model_on = register_model_on
        self.log_model_at_end_only = log_model_at_end_only
        self._last_model_uri = None
        self.last_saved_model_info = None
        self._inferred_signature = None
        self._input_example = None
        self.code_paths = code_paths
        self.additional_metadata = additional_metadata or {}
        self.extra_pip_requirements = extra_pip_requirements or []

    def _infer_params(self, model: nn.Module) -> tuple[dict, ...]:
        """Extract metadata from the model's forward method signature.

        Returns:
            A tuple of dicts, each containing parameter metadata ordered by position.
        """
        forward_method = getattr(model.__class__, 'forward', None)

        if forward_method is None:
            return ()

        try:
            sig = inspect.signature(forward_method)
            params_list = []

            for param_name, param in sig.parameters.items():
                if param_name == 'self':
                    continue

                param_info = {
                    'name': param_name,
                    'kind': param.kind.name,
                    'annotation': param.annotation if param.annotation != inspect.Parameter.empty else None,
                    'default': param.default if param.default != inspect.Parameter.empty else None,
                }
                params_list.append(param_info)

            # Add return annotation if available as the last element
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                return_info = {'_return_annotation': str(return_annotation)}
                params_list.append(return_info)

            return tuple(params_list)

        except Exception as e:
            _LOGGER.warning(f"Failed to infer forward method parameters: {e}")
            return ()

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        _LOGGER.debug(f"Saving checkpoint to {filepath}...")
        trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
                if isinstance(logger, MLFlowLogger) and not self.log_model_at_end_only:
                    _LOGGER.debug(f"_save_checkpoint: Logging model to MLFlow at {filepath}...")
                    self.log_model_to_mlflow(trainer.model, run_id=logger.run_id)

    def log_additional_metadata(self, logger: MLFlowLogger | L.Trainer,
                                additional_metadata: dict) -> None:
        """Log additional metadata as a JSON file to the model artifact.

        Args:
            logger: The MLFlowLogger or Lightning Trainer instance to use for logging.
            additional_metadata: A dictionary containing additional metadata to log.
        """
        self.additional_metadata = additional_metadata
        if not self.additional_metadata:
            return

        if self.last_saved_model_info is None:
            _LOGGER.warning("No model has been saved yet. Cannot log additional metadata.")
            return

        try:
            log_model_metadata(metadata=self.additional_metadata,
                               logger=logger,
                               model_path=self.last_saved_model_info.artifact_path)
        except Exception as e:
            _LOGGER.warning(f"Failed to log additional metadata: {e}")

    def log_model_to_mlflow(self,
                            model: nn.Module,
                            run_id: str | MLFlowLogger
                            ) -> None:
        """Log the model to MLflow."""
        if isinstance(run_id, MLFlowLogger):
            logger = run_id
            if logger.run_id is None:
                raise ValueError("MLFlowLogger has no run_id. Cannot log model to MLFlow.")
            run_id = logger.run_id

        if self._last_checkpoint_saved is None or self._last_checkpoint_saved == '':
            _LOGGER.warning("No checkpoint saved yet. Cannot log model to MLFlow.")
            return

        orig_device = next(model.parameters()).device
        model = model.cpu()  # Ensure the model is on CPU for logging

        requirements = list(self.extra_pip_requirements)
        # check if lightning is in the requirements
        if not any('lightning' in req.lower() for req in requirements):
            requirements.append(f'lightning=={L.__version__}')

        _LOGGER.debug(f"log_model_to_mlflow: Logging model to MLFlow at {self._last_checkpoint_saved}...")
        modelinfo = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path=f'model/{Path(self._last_checkpoint_saved).stem}',
            signature=self._inferred_signature,
            run_id=run_id,
            extra_pip_requirements=requirements,
            code_paths=self.code_paths
        )

        model.to(device=orig_device)  # Move the model back to its original device
        self._last_model_uri = modelinfo.model_uri
        self.last_saved_model_info = modelinfo

        # Log additional metadata after the model is saved
        log_model_metadata(self.additional_metadata,
                           model_path=modelinfo.artifact_path,
                           run_id=run_id)

    def _remove_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        # remove the checkpoint from mlflow
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                if isinstance(logger, MLFlowLogger):
                    artifact_uri = logger.experiment.get_run(logger.run_id).info.artifact_uri
                    rep = get_artifact_repository(artifact_uri)
                    rep.delete_artifacts(f'model/{Path(filepath).stem}')

    def register_model(self, trainer=None):
        """Register the model in MLFlow Model Registry."""
        # mlflow_client = _get_MLFlowLogger(trainer)._mlflow_client
        return mlflow.register_model(
            model_uri=self._last_model_uri,
            name=self.register_model_name,
        )

    def _update_signature(self, trainer):
        if self._inferred_signature is None:
            _LOGGER.warning("No signature found. Cannot update signature.")
            return
        if self._last_model_uri is None:
            _LOGGER.warning("No model URI found. Cannot update signature.")
            return

        mllogger = _get_MLFlowLogger(trainer)
        mlclient = mllogger._mlflow_client

        # check if the model exists
        for artifact_info in mlclient.list_artifacts(run_id=mllogger.run_id):
            if artifact_info.path.startswith('model'):
                break
        else:
            _LOGGER.warning(f"Model URI {self._last_model_uri} does not exist. Cannot update signature.")
            return
        _LOGGER.debug(f"Updating signature for model URI: {self._last_model_uri}...")
        # update the signature
        mlflow.models.set_signature(
            model_uri=self._last_model_uri,
            signature=self._inferred_signature,
        )

    def __wrap_forward(self, pl_module: nn.Module):
        original_forward = pl_module.forward

        def wrapped_forward(x, *args, **kwargs):
            x0 = help_infer_signature(x)
            infered_params = self._infer_params(pl_module)
            if len(infered_params) > 1:
                infered_params = {param['name']: param['default']
                                  for param in infered_params[1:] if 'name' in param}
            else:
                infered_params = None

            self._inferred_signature = mlflow.models.infer_signature(model_input=x0,
                                                                     params=infered_params)


            # run once and get back to the original forward
            pl_module.forward = original_forward
            method = getattr(pl_module, 'forward')
            out = method(x, *args, **kwargs)

            output_sig = mlflow.models.infer_signature(model_output=help_infer_signature(out))
            self._inferred_signature.outputs = output_sig.outputs

            return out

        pl_module.forward = wrapped_forward

    def on_train_start(self, trainer, pl_module):
        self.__wrap_forward(pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_train_end(trainer, pl_module)

        if self.log_model_at_end_only and trainer.is_global_zero:
            logger = _get_MLFlowLogger(trainer)
            if logger is None:
                _LOGGER.warning("No MLFlowLogger found. Cannot log model to MLFlow.")
            else:
                self.log_model_to_mlflow(trainer.model, run_id=logger.run_id)

        self._update_signature(trainer)

        if self.register_model_on == 'train':
            self.register_model(trainer)

    def _restore_model_uri(self, trainer: L.Trainer) -> None:
        logger = _get_MLFlowLogger(trainer)
        if logger is None:
            _LOGGER.warning("No MLFlowLogger found. Cannot restore model URI.")
            return
        if trainer.ckpt_path is None:
            return
        extracted_run_id = Path(trainer.ckpt_path).parts[1]
        if extracted_run_id != logger.run_id:
            _LOGGER.warning(f"Run ID mismatch: {extracted_run_id} != {logger.run_id}." +
                            " Check `run_id` parameter in MLFlowLogger.")
        self._last_model_uri = f'runs:/{logger.run_id}/model/{Path(trainer.ckpt_path).stem}'
        try:
            self.last_saved_model_info = mlflow.models.get_model_info(self._last_model_uri)
        except mlflow.exceptions.MlflowException as e:
            _LOGGER.warning(f"Failed to get model info for URI {self._last_model_uri}: {e}")
            self.last_saved_model_info = None

    def on_test_start(self, trainer, pl_module):
        self.__wrap_forward(pl_module)
        self._restore_model_uri(trainer)
        return super().on_test_start(trainer, pl_module)

    def on_predict_start(self, trainer, pl_module):
        self.__wrap_forward(pl_module)
        self._restore_model_uri(trainer)
        return super().on_predict_start(trainer, pl_module)

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_test_end(trainer, pl_module)

        if self.register_model_on == 'test':
            self._update_signature(trainer)
            self.register_model(trainer)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_predict_end(trainer, pl_module)

        if self.register_model_on == 'predict':
            self._update_signature(trainer)
            self.register_model(trainer)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)

        if self.register_model_on == 'val':
            self._update_signature(trainer)
            self.register_model(trainer)
