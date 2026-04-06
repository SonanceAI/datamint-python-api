from collections.abc import Mapping
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path
from weakref import proxy
from mlflow.store.artifact.artifact_repository_registry import get_artifact_repository
from typing import Literal, Any, TYPE_CHECKING
import inspect
from torch import nn
import lightning.pytorch as L
from datamint.mlflow.models import log_model_metadata, _get_MLFlowLogger
from datamint.mlflow.env_utils import ensure_mlflow_configured
import mlflow.models
import mlflow.exceptions
import mlflow.pytorch
import mlflow.data.dataset
import mlflow.entities.dataset
import copy
import logging
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, Future
from lightning.pytorch.loggers import MLFlowLogger

if TYPE_CHECKING:
    from datamint.mlflow.flavors.model import BaseDatamintModel
    from mlflow.models.model import ModelInfo

_LOGGER = logging.getLogger(__name__)


def _prepare_signature_sample(x: Any) -> Any:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, Mapping):
        return {k: _prepare_signature_sample(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [_prepare_signature_sample(v) for v in x]
    elif isinstance(x, tuple):
        return tuple(_prepare_signature_sample(v) for v in x)
    return x


class _BaseMLFlowModelCheckpoint(ModelCheckpoint):
    """Base class for MLflow-integrated model checkpoint callbacks.

    Provides all shared logic for checkpointing, model registration, signature updates,
    and metric logging. Subclasses must implement :meth:`log_model_to_mlflow` and
    :meth:`_wrap_forward` for their specific MLflow flavor and signature-inference strategy.
    """

    def __init__(self, *args,
                 register_model_name: str | None = None,
                 register_model_on: Literal["train", "val", "test", "predict"] = 'test',
                 code_paths: list[str] | None = None,
                 log_model_at_end_only: bool = True,
                 additional_metadata: dict[str, Any] | None = None,
                 extra_pip_requirements: list[str] | None = None,
                 log_model_metrics: bool = True,
                 **kwargs):
        """
        Args:
            register_model_name (str | None): The name to register the model under in MLFlow.
                If None, the model will not be registered.
            register_model_on (Literal["train", "val", "test", "predict"]): The stage at which
                to register the model. It registers at the end of the specified stage.
            code_paths (list[str] | None): List of paths to Python files that should be
                included in the MLFlow model.
            log_model_at_end_only (bool): If True, only log the model to MLFlow at the end of
                training instead of after every checkpoint save.
            additional_metadata (dict[str, Any] | None): Additional metadata to log with the
                model as a JSON file.
            extra_pip_requirements (list[str] | None): Additional pip requirements to include
                with the MLFlow model.
            log_model_metrics (bool): If True, automatically log test metrics to the MLflow
                LoggedModel entity after testing. Requires MLflow 3.x. Defaults to True.
            **kwargs: Keyword arguments for ModelCheckpoint.
        """
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
        if register_model_on not in ["train", "val", "test", "predict"]:
            raise ValueError("register_model_on must be one of train, val, test or predict.")

        self.register_model_name = register_model_name
        self.register_model_on = register_model_on
        self.registered_model_info = None
        self.log_model_at_end_only = log_model_at_end_only
        self.log_model_metrics = log_model_metrics
        self._last_model_uri = None
        self._last_model_id: str | None = None
        self.last_saved_model_info = None
        self._inferred_signature = None
        self.code_paths = code_paths
        self.additional_metadata = additional_metadata or {}
        self.extra_pip_requirements = extra_pip_requirements or []
        self._last_registered_state_hash: str | None = None
        self._has_been_trained: bool = False
        self._signature_forward_wrapped: bool = False
        self._logging_executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix='mlflow-log')
        self._logging_future: Future | None = None
        self._pending_log_model: nn.Module | None = None

    def _wait_for_pending_logging(self) -> None:
        """Block until any in-flight background model logging completes."""
        if self._logging_future is None:
            return
        try:
            self._logging_future.result()
        except Exception:
            _LOGGER.exception("Background model logging failed")
        finally:
            self._logging_future = None
        # Inject model_id into the original training model
        if self._pending_log_model is not None:
            self._inject_model_id(self._pending_log_model)
            self._pending_log_model = None

    def _inject_model_id(self, model: 'nn.Module | L.LightningModule | BaseDatamintModel') -> None:
        """Inject the MLflow model ID into the model, if it supports it."""
        if self._last_model_id is None:
            return
        if hasattr(model, 'set_mlflow_model_id'):
            model.set_mlflow_model_id(self._last_model_id)
        elif hasattr(model, 'mlflow_model_id'):
            setattr(model, 'mlflow_model_id', self._last_model_id)

    def _prepare_loggable_model(self, model: nn.Module) -> nn.Module:
        """Prepare a model for MLflow logging, potentially creating a CPU copy.

        Called on the main thread before async logging.
        Override in subclasses that need thread-safe model snapshots.
        Returns the same model by default (sync logging).
        """
        return model

    def get_last_model_id(self) -> str | None:
        """Get the MLflow model ID of the last saved model, if available."""
        self._wait_for_pending_logging()
        return self._last_model_id

    def get_last_model_uri(self) -> str | None:
        """Get the MLflow model URI of the last saved model, if available."""
        self._wait_for_pending_logging()
        return self._last_model_uri

    def get_all_saved_models(self):
        """Get a list of all MLflow ModelInfo objects for models logged from this callback."""
        self._wait_for_pending_logging()
        if self._last_model_uri is None:
            return []
        logger = _get_MLFlowLogger()
        if logger is None or logger.run_id is None:
            _LOGGER.warning("No MLFlowLogger run_id found. Cannot retrieve saved models.")
            return []
        try:
            retrieved_logged_models = mlflow.search_logged_models(
                filter_string=f"name = '{Path(self._last_model_uri).stem[:256]}' AND source_run_id='{logger.run_id[:64]}'",
                order_by=[{"field_name": "last_updated_timestamp", "ascending": False}],
                output_format="list"
            )
            return retrieved_logged_models
        except Exception as e:
            _LOGGER.warning(f"Failed to retrieve saved models: {e}")
            return []

    def _compute_registration_state_hash(self) -> str:
        """Compute a hash representing the current model state for registration comparison.

        Returns:
            A hash string of the current state, or None if state cannot be computed.
        """
        state_dict = {
            'checkpoint_path': str(self._last_checkpoint_saved),
            'global_step': self._last_global_step_saved,
            'signature': str(self._inferred_signature) if self._inferred_signature else None,
            'model_uri': self._last_model_uri,
        }

        state_str = json.dumps(state_dict, sort_keys=True)
        return hashlib.md5(state_str.encode('utf-8')).hexdigest()

    def _should_register_model(self) -> bool:
        """Determine if the model should be registered.

        Returns:
            True if the model should be registered, False otherwise.
        """

        if self._last_model_uri is None:
            _LOGGER.warning("No model URI available. Cannot register model.")
            return False

        # If never registered before, register
        if self._last_registered_state_hash is None:
            return True

        # If model was retrained, register
        if self._has_been_trained:
            return True

        # If state changed (signature, checkpoint, etc.), register
        current_state_hash = self._compute_registration_state_hash()
        if current_state_hash != self._last_registered_state_hash:
            return True

        _LOGGER.info("Model already registered with same configuration. Skipping registration.")
        return False

    def _save_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        self._wait_for_pending_logging()
        trainer.save_checkpoint(filepath, self.save_weights_only)

        self._last_global_step_saved = trainer.global_step
        self._last_checkpoint_saved = filepath

        # notify loggers
        if trainer.is_global_zero:
            for logger in trainer.loggers:
                logger.after_save_checkpoint(proxy(self))
                if isinstance(logger, MLFlowLogger) and not self.log_model_at_end_only:
                    loggable = self._prepare_loggable_model(trainer.model)
                    if loggable is not trainer.model:
                        # Snapshot created; safe to log in background thread
                        self._pending_log_model = trainer.model
                        self._logging_future = self._logging_executor.submit(
                            self.log_model_to_mlflow, loggable, logger.run_id
                        )
                    else:
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
                            model: 'nn.Module | L.LightningModule | BaseDatamintModel',
                            run_id: str | MLFlowLogger) -> None:
        """Log the model to MLflow using the appropriate flavor.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def _remove_checkpoint(self, trainer: L.Trainer, filepath: str) -> None:
        self._wait_for_pending_logging()
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
        self._wait_for_pending_logging()
        if not self._should_register_model():
            return self.registered_model_info

        # mlflow_client = _get_MLFlowLogger(trainer)._mlflow_client
        self.registered_model_info = mlflow.register_model(
            model_uri=self._last_model_uri,
            name=self.register_model_name,
        )

        # Update the registered state hash after successful registration
        self._last_registered_state_hash = self._compute_registration_state_hash()
        self._has_been_trained = False  # Reset training flag after registration

        _LOGGER.info(f"Model registered as '{self.register_model_name}' "
                     f"version {self.registered_model_info.version}")

        return self.registered_model_info

    def _update_signature(self, trainer):
        self._wait_for_pending_logging()
        if self._inferred_signature is None:
            _LOGGER.warning("No signature found. Cannot update signature.")
            return
        if self._last_model_uri is None:
            _LOGGER.warning("No model URI found. Cannot update signature.")
            return

        # update the signature
        try:
            mlflow.models.set_signature(
                model_uri=self._last_model_uri,
                signature=self._inferred_signature,
            )
        except mlflow.exceptions.MlflowException as e:
            _LOGGER.warning(f"Failed to update model signature. Check if model actually exists. {e}")

    def _resolve_run_id(self, run_id: str | MLFlowLogger) -> str:
        """Extract the run_id string from an MLFlowLogger or pass through a string."""
        if isinstance(run_id, MLFlowLogger):
            if run_id.run_id is None:
                raise ValueError("MLFlowLogger has no run_id. Cannot log model to MLFlow.")
            return run_id.run_id
        return run_id

    def _build_requirements(self) -> list[str]:
        """Build pip requirements list, ensuring lightning is included."""
        requirements = list(self.extra_pip_requirements)
        if not any('lightning' in req.lower() for req in requirements):
            requirements.append(f'lightning=={L.__version__}')
        return requirements

    def _finalize_logged_model(self,
                               modelinfo: 'ModelInfo',
                               run_id: str,
                               model: 'nn.Module | L.LightningModule | BaseDatamintModel') -> None:
        """Store model info and log metadata after logging a model to MLflow."""
        self._last_model_uri = modelinfo.model_uri
        self._last_model_id = getattr(modelinfo, 'model_id', None)
        self.last_saved_model_info = modelinfo
        _LOGGER.debug("Model logged to MLflow with URI: %s and ID: %s", self._last_model_uri, self._last_model_id)
        if self.additional_metadata:
            _LOGGER.debug("Logging additional metadata for model %s with run_id %s: %s",
                          modelinfo.model_uri, run_id, self.additional_metadata)
            log_model_metadata(self.additional_metadata,
                               model_path=modelinfo.artifact_path,
                               run_id=run_id)

        self._inject_model_id(model)
        _LOGGER.debug("Finalized logged model with ID %s and URI %s", self._last_model_id, self._last_model_uri)

    def _wrap_forward(self, pl_module: nn.Module) -> None:
        """Intercept the first forward call to infer the MLflow model signature.

        Override in subclasses to customize signature inference.
        """
        pass

    def on_train_start(self, trainer, pl_module):
        self._has_been_trained = True
        self._wrap_forward(pl_module)
        logger = _get_MLFlowLogger(trainer)
        if logger._tracking_uri.startswith('file:'):
            _LOGGER.error("MLFlowLogger tracking URI is a local file path. "
                          "Model registration will likely fail if using MLflow Model Registry.")
        if logger.experiment_id is not None:
            mlflow.set_experiment(experiment_id=logger.experiment_id)
        super().on_train_start(trainer, pl_module)

    def on_train_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_train_end(trainer, pl_module)
        self._wait_for_pending_logging()

        if self.log_model_at_end_only and trainer.is_global_zero:
            logger = _get_MLFlowLogger(trainer)
            self.log_model_to_mlflow(trainer.model, run_id=logger.run_id)

        self._update_signature(trainer)

        if self.register_model_on == 'train' and self.register_model_name:
            self.register_model(trainer)

    def _restore_model_uri(self, trainer: L.Trainer) -> None:
        """Restore the last model URI from the trainer's checkpoint path.
        """
        logger = _get_MLFlowLogger(trainer)
        self._last_model_uri = None
        self._last_model_id = None
        self.last_saved_model_info = None
        if logger is None:
            _LOGGER.warning("No MLFlowLogger found. Cannot restore model URI.")
            return
        if trainer.ckpt_path is None:
            return
        if logger.run_id is None:
            _LOGGER.warning("MLFlowLogger has no run_id. Cannot restore model URI.")
            return
        if logger.run_id not in str(trainer.ckpt_path):
            _LOGGER.warning(f"Run ID mismatch between checkpoint path and MLFlowLogger." +
                            " Check `run_id` parameter in MLFlowLogger.")
            return
        model_name = Path(trainer.ckpt_path).stem[:256]
        retrieved_logged_models = mlflow.search_logged_models(
            filter_string=f"name = '{model_name}' AND source_run_id='{logger.run_id[:64]}'",
            order_by=[{"field_name": "last_updated_timestamp", "ascending": False}],
            output_format="list"
        )
        if not retrieved_logged_models:
            _LOGGER.warning(f"No logged model found for checkpoint {model_name}.")
            return
        # get the most recent one
        self._last_model_uri = retrieved_logged_models[0].model_uri
        self._last_model_id = getattr(retrieved_logged_models[0], 'model_id', None)
        try:
            self.last_saved_model_info = mlflow.models.get_model_info(self._last_model_uri)
        except mlflow.exceptions.MlflowException as e:
            _LOGGER.warning(f"Failed to get model info for URI {self._last_model_uri}: {e}")
            self.last_saved_model_info = None

    def on_test_start(self, trainer, pl_module):
        self._wait_for_pending_logging()
        self._wrap_forward(pl_module)
        self._restore_model_uri(trainer)
        return super().on_test_start(trainer, pl_module)

    def on_predict_start(self, trainer, pl_module):
        self._wait_for_pending_logging()
        self._wrap_forward(pl_module)
        self._restore_model_uri(trainer)
        return super().on_predict_start(trainer, pl_module)

    def _log_test_metrics_to_model(self, trainer: L.Trainer) -> None:
        """Log test metrics from trainer.callback_metrics to the MLflow LoggedModel.

        Filters metrics to only include those prefixed with 'test/' or 'test_',
        converts tensor values to floats, and logs them to the LoggedModel
        identified by ``self._last_model_id``.
        """
        # TODO: Separate this into its own callback that depends on the model checkpoint callback or a injection of a model_id.
        # Also consider logging all metrics with a model_id tag instead of just test metrics, and allowing users to configure the prefix filter.
        if self._last_model_id is None:
            _LOGGER.debug("No model_id available. Skipping model metrics logging.")
            return

        logger = _get_MLFlowLogger(trainer)
        if logger is None or logger.run_id is None:
            _LOGGER.warning("No MLFlowLogger run_id found. Skipping model metrics logging.")
            return

        metrics: dict[str, float] = {}
        for key, value in trainer.callback_metrics.items():
            if not key.startswith(("test/", "test_")):
                continue
            try:
                metrics[key] = float(value)
            except (TypeError, ValueError):
                _LOGGER.debug(f"Skipping non-numeric metric '{key}': {value}")

        if not metrics:
            _LOGGER.info("No test metrics found in callback_metrics to log.")
            return

        dataset = getattr(logger, '_mlflow_dataset', None)
        if not isinstance(dataset, (mlflow.data.dataset.Dataset, mlflow.entities.dataset.Dataset)):
            dataset = None
            _LOGGER.warning(
                "Logger dataset is not an MLflow Dataset. Proceeding without dataset context for metrics logging.")
        try:
            mlflow.log_metrics(metrics,
                               model_id=self._last_model_id,
                               run_id=logger.run_id,
                               dataset=dataset)
            _LOGGER.info(f"Logged {len(metrics)} test metrics to model {self._last_model_id}.")
        except Exception as e:
            _LOGGER.warning(f"Failed to log test metrics to model: {e}")

    def on_test_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_test_end(trainer, pl_module)

        if self.log_model_metrics:
            self._log_test_metrics_to_model(trainer)

        if self.register_model_on == 'test' and self.register_model_name:
            self._update_signature(trainer)
            self.register_model(trainer)

    def on_predict_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_predict_end(trainer, pl_module)

        if self.register_model_on == 'predict' and self.register_model_name:
            self._update_signature(trainer)
            self.register_model(trainer)

    def on_validation_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        super().on_validation_end(trainer, pl_module)

        if self.register_model_on == 'val' and self.register_model_name:
            self._update_signature(trainer)
            self.register_model(trainer)


class MLFlowPyTorchModelCheckpoint(_BaseMLFlowModelCheckpoint):
    """MLflow model checkpoint for standard PyTorch Lightning modules.

    Logs models using :func:`mlflow.pytorch.log_model` and infers the MLflow
    model signature by intercepting the first call to ``pl_module.forward``.
    """

    def _infer_forward_defaults(self, model: nn.Module) -> dict[str, Any] | None:
        """Extract default values for forward() params beyond the first input.

        Returns:
            dict of {name: default} or None if no extra params exist.
        """
        forward_method = getattr(model.__class__, 'forward', None)
        if forward_method is None:
            return None

        try:
            sig = inspect.signature(forward_method)
            params = [p for name, p in sig.parameters.items() if name != 'self']
            # Skip the first input parameter, collect defaults for the rest
            defaults = {
                p.name: (p.default if p.default != inspect.Parameter.empty else None)
                for p in params[1:]
            }
            return defaults or None
        except Exception as e:
            _LOGGER.warning(f"Failed to infer forward method parameters: {e}")
            return None

    def _wrap_forward(self, pl_module: nn.Module) -> None:
        """Wrap ``pl_module.forward`` to infer the MLflow signature on the first call."""
        if self._inferred_signature is not None:
            return
        if self._signature_forward_wrapped:
            return

        original_forward = pl_module.forward
        self._signature_forward_wrapped = True

        def wrapped_forward(x, *args, **kwargs):
            self._inferred_signature = mlflow.models.infer_signature(
                model_input=_prepare_signature_sample(x),
                params=self._infer_forward_defaults(pl_module),
            )

            # Restore original forward and call it
            pl_module.forward = original_forward
            self._signature_forward_wrapped = False
            out = original_forward(x, *args, **kwargs)

            output_sig = mlflow.models.infer_signature(model_output=_prepare_signature_sample(out))
            self._inferred_signature.outputs = output_sig.outputs

            return out

        pl_module.forward = wrapped_forward

    def _prepare_loggable_model(self, model: nn.Module) -> nn.Module:
        """Create a CPU deep copy for thread-safe background logging.

        The training model remains on its original device, avoiding
        a costly GPU synchronisation and host-to-device transfer.
        """
        import torch
        with torch.no_grad():
            cpu_model = copy.deepcopy(model)
            cpu_model.cpu()
        return cpu_model

    def log_model_to_mlflow(self,
                            model: 'nn.Module | L.LightningModule | BaseDatamintModel',
                            run_id: str | MLFlowLogger):
        """Log the model to MLflow using the pytorch flavor."""
        run_id = self._resolve_run_id(run_id)

        if run_id is None:
            _LOGGER.warning("No run_id available from the logger. Skipping MLflow model logging "
                            "to avoid creating a new run.")
            return

        if not self._last_checkpoint_saved:
            _LOGGER.warning("No checkpoint saved yet. Cannot log model to MLFlow.")
            return

        import torch

        # If model is on a GPU, create a CPU copy instead of moving the
        # training model off-device, avoiding a costly sync + H2D transfer.
        device = next(model.parameters()).device
        if device.type != 'cpu':
            with torch.no_grad():
                model_to_log = copy.deepcopy(model)
                model_to_log.cpu()
        else:
            model_to_log = model

        _LOGGER.info("Logging model using pytorch flavor with name %s", Path(self._last_checkpoint_saved).stem)
        modelinfo = mlflow.pytorch.log_model(
            pytorch_model=model_to_log,
            name=Path(self._last_checkpoint_saved).stem,
            signature=self._inferred_signature,
            run_id=run_id,
            extra_pip_requirements=self._build_requirements(),
            code_paths=self.code_paths,
        )

        if model_to_log is not model:
            del model_to_log
        self._finalize_logged_model(modelinfo, run_id, model=model)


class MLFlowDatamintModelCheckpoint(_BaseMLFlowModelCheckpoint):
    """MLflow model checkpoint for :class:`~datamint.mlflow.flavors.model.BaseDatamintModel`-based
    Lightning modules.

    Logs models using the datamint custom flavor (which wraps ``mlflow.pyfunc``).
    Signature inference is delegated to the datamint flavor via ``predict_type_hints``,
    so no forward-wrapping is performed.
    """

    def log_model_to_mlflow(self,
                            model: 'nn.Module | L.LightningModule | BaseDatamintModel',
                            run_id: str | MLFlowLogger) -> None:
        """Log the model to MLflow using the datamint flavor."""
        run_id = self._resolve_run_id(run_id)

        if run_id is None:
            _LOGGER.warning("No run_id available from the logger. Skipping MLflow model logging "
                            "to avoid creating a new run.")
            return

        if not self._last_checkpoint_saved:
            _LOGGER.warning("No checkpoint saved yet. Cannot log model to MLFlow.")
            return

        from datamint.mlflow.flavors import datamint_flavor
        _LOGGER.info("Logging model using datamint flavor with name %s", Path(self._last_checkpoint_saved).stem)
        modelinfo = datamint_flavor.log_model(
            model,
            name=Path(self._last_checkpoint_saved).stem,
            signature=self._inferred_signature,
            run_id=run_id,
            extra_pip_requirements=self._build_requirements(),
            code_paths=self.code_paths,
        )

        self._finalize_logged_model(modelinfo, run_id, model=model)

    def _update_signature(self, trainer):
        return  # signature is managed by the datamint flavor, so we don't need to do anything here


# Backward-compatibility alias
MLFlowModelCheckpoint = MLFlowPyTorchModelCheckpoint
