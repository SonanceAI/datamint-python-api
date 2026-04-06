"""
DataMint Model Adapter Module

This module provides a flexible framework for wrapping ML models to work with DataMint's
annotation system. It supports various prediction modes for different data types and use cases.
"""

from typing import Any, ClassVar, TypeAlias
from abc import ABC
from dataclasses import dataclass
from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext
from datamint.entities.annotations import Annotation
from datamint.entities.resource import Resource
from datamint.mlflow.flavors.model_loader import LinkedModelLoader
from datamint.mlflow.flavors.prediction_modes import PredictionMode
from datamint.mlflow.flavors.task_type import TaskType
from datamint.mlflow.flavors.prediction_router import PredictionRouter
import logging
from functools import cached_property
import torch
from mlflow.pytorch import pickle_module as mlflow_pytorch_pickle_module

logger = logging.getLogger(__name__)

# Type aliases
PredictionResult: TypeAlias = list[list[Annotation]]


@dataclass
class ModelSettings:
    """
    Deployment and inference configuration for DatamintModel.

    These settings are serialized with the model and used by remote MLflow servers
    to properly configure the runtime environment.
    """
    # Hardware requirements
    need_gpu: bool = False
    """Whether GPU is required for inference"""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> 'ModelSettings':
        """Create config from dictionary, raising error on unknown keys."""
        valid_fields = set(cls.__dataclass_fields__)
        invalid_fields = set(data.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid fields for ModelSettings: {', '.join(sorted(invalid_fields))}")
        return cls(**data)


class BaseDatamintModel(PythonModel, ABC):
    """Core prediction gateway that any MLflow :class:`~mlflow.pyfunc.PythonModel` can build on.

    Owns:

    * :attr:`settings` — hardware / deployment configuration.
    * Device detection (``_detect_device`` / :attr:`inference_device`).
    * Prediction dispatch via :class:`~datamint.mlflow.flavors.prediction_router.PredictionRouter`.
    * Pickle-safe serialization.

    Use :class:`DatamintModel` when you need to load linked models at serve time.

    Subclasses only need to implement :meth:`predict_default` (and optionally
    other ``predict_*`` hooks registered with ``@prediction_mode``).
    """

    task_type: ClassVar[TaskType | None] = None
    """Semantic task category for this model class. Subclasses should override
    at the class body level (e.g. ``task_type = TaskType.IMAGE_SEGMENTATION``)."""

    def __init__(
        self,
        settings: ModelSettings | dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self._inference_device: str | None = None

    @cached_property
    def _router(self) -> PredictionRouter:
        """The PredictionRouter instance responsible for dispatching predict calls."""
        return PredictionRouter(self, BaseDatamintModel)

    @property
    def settings(self) -> ModelSettings:
        if not hasattr(self, "_settings"):
            self._settings = ModelSettings()
        return self._settings

    @settings.setter
    def settings(self, value: ModelSettings | dict[str, Any] | None) -> None:
        if isinstance(value, dict):
            self._settings = ModelSettings.from_dict(value)
        elif isinstance(value, ModelSettings):
            self._settings = value
        else:
            self._settings = ModelSettings()

    # ------------------------------------------------------------------
    # Device management
    # ------------------------------------------------------------------

    @property
    def inference_device(self) -> str:
        """The device that will be used for inference.

        Returns ``_inference_device`` if already set, then falls back to the
        ``MLFLOW_DEFAULT_PREDICTION_DEVICE`` environment variable, then ``'cpu'``.
        """
        if self._inference_device is not None:
            return self._inference_device
        env_device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
        if env_device:
            logger.info("Inference device not set; using environment variable (%s)", env_device)
            return env_device
        logger.warning("Inference device not set; defaulting to 'cpu'")
        return "cpu"

    def _detect_device(self, context: PythonModelContext | None) -> str:
        """Detect and store the inference device from context / env / hardware.

        Sets :attr:`_inference_device` and returns the detected device string.
        Priority: ``context.model_config['device']`` > env var > CUDA > CPU.
        """
        device = None
        if context and context.model_config:
            device = context.model_config.get("device", None)
            logger.info("Model config device: %s", device)
        if device is None:
            env_device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
            if env_device:
                device = env_device
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        logger.info("Set inference device: %s", device)
        self._inference_device = device
        return device

    # ------------------------------------------------------------------
    # MLflow lifecycle
    # ------------------------------------------------------------------

    def load_context(self, context: PythonModelContext) -> None:
        """Detect the inference device.

        Override in subclasses to perform additional loading (e.g. linked
        models) — but always call ``super().load_context(context)`` first
        so that :attr:`inference_device` is set before any model loading.
        """
        logger.info("Loading model context %s and detecting device...",
                    f'{context.artifacts=} | {context.model_config=}')
        self._detect_device(context)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_router", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Prediction dispatch
    # ------------------------------------------------------------------
    def predict(
        self,
        model_input: list[Resource],
        params: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Main prediction entry point.

        Routes to the appropriate handler based on ``params['mode']``.
        **Do not override** — implement :meth:`predict_default` (or other
        ``predict_*`` hooks) instead.
        """
        return self._router.dispatch(model_input, params or {})

    def get_supported_modes(self) -> list[str]:
        """Return the list of prediction modes supported by this model."""
        return self._router.supported_modes()

    def predict_default(
        self,
        model_input: list[Resource],
        **kwargs: Any,
    ) -> PredictionResult:
        """Default prediction on entire resources.

        Override this in your subclass.
        """
        raise NotImplementedError(
            "predict_default() must be implemented in your DatamintModel subclass."
        )


class DatamintModel(BaseDatamintModel):
    """Abstract adapter for wrapping ML models to produce Datamint annotations.

    Extends :class:`BaseDatamintModel` with support for loading external
    ("linked") MLflow models at serve time via :class:`LinkedModelLoader`.
    Subclasses only need to override ``predict_default`` (and optionally
    other ``predict_*`` hooks).

    Quick Start::

        class MyModel(DatamintModel):
            def __init__(self):
                super().__init__(
                    mlflow_models_uri={'model': 'models:/MyModel/latest'},
                    settings=ModelSettings(need_gpu=True),
                )

            def predict_default(self, model_input, **kwargs):
                device = self.inference_device
                model = self.get_mlflow_models()['model'].get_raw_model().to(device)
                return predictions

    You can also pass pre-instantiated ``torch.nn.Module`` objects directly::

        class MyModel(DatamintModel):
            def __init__(self):
                net = MyTorchNet()
                super().__init__(torch_model=net)

            def predict_default(self, model_input, **kwargs):
                net = self.get_mlflow_torch_models()['net']
                return net(preprocess(model_input))
    """

    # Keep for backward compat with subclass references
    LINKED_MODELS_DIR = "linked_models"
    _PYTORCH_ARTIFACT_NAME = "pytorch_model"

    def __init__(
        self,
        settings: ModelSettings | dict[str, Any] | None = None,
        mlflow_torch_models_uri: dict[str, str] | None = None,
        mlflow_models_uri: dict[str, str] | None = None,
        torch_model: torch.nn.Module | None = None,
    ) -> None:
        super().__init__(settings=settings)
        self._loader = LinkedModelLoader(
            mlflow_models_uri=mlflow_models_uri,
            mlflow_torch_models_uri=mlflow_torch_models_uri,
            torch_model=torch_model,
        )

    @cached_property
    def _router(self) -> PredictionRouter:
        """The PredictionRouter instance responsible for dispatching predict calls."""
        r = PredictionRouter(self, BaseDatamintModel)
        if self._loader._torch_model_instance is not None:
            r.update_registry(self._loader._torch_model_instance, BaseDatamintModel)
        return r

    # ------------------------------------------------------------------
    # Lifecycle — overrides base to also load linked models
    # ------------------------------------------------------------------

    def load_context(self, context: PythonModelContext) -> None:
        """Detect device and load all linked MLflow models."""
        super().load_context(context)  # sets inference_device
        self._loader.load_all(self.inference_device)

        if self._PYTORCH_ARTIFACT_NAME in context.artifacts:
            model_path = context.artifacts[self._PYTORCH_ARTIFACT_NAME]
            self._loader._torch_model_instance = torch.load(model_path,
                                                            weights_only=False,
                                                            map_location=self.inference_device,
                                                            pickle_module=mlflow_pytorch_pickle_module)
            self._loader._torch_model_instance.eval()

    # ------------------------------------------------------------------
    # Linked-model access
    # ------------------------------------------------------------------

    def get_mlflow_models(self) -> dict[str, PyFuncModel]:
        """Access loaded MLflow pyfunc models."""
        return self._loader.mlflow_models

    def get_mlflow_torch_models(self) -> dict[str, Any]:
        """Access loaded MLflow PyTorch models."""
        return self._loader.torch_models

    def get_pytorch_model(self) -> torch.nn.Module | None:
        torch_model = self._loader._torch_model_instance
        if torch_model is not None:
            return torch_model
        torch_models = self.get_mlflow_torch_models()
        if len(torch_models) == 1:
            return next(iter(torch_models.values()))
        return torch_models.get("default", None)

    # ------------------------------------------------------------------
    # Backward-compat aliases
    # ------------------------------------------------------------------

    @property
    def mlflow_models_uri(self) -> dict[str, str]:
        return self._loader.mlflow_models_uri

    @mlflow_models_uri.setter
    def mlflow_models_uri(self, value: dict[str, str]) -> None:
        self._loader.mlflow_models_uri = value

    @property
    def mlflow_torch_models_uri(self) -> dict[str, str]:
        return self._loader.mlflow_torch_models_uri

    @mlflow_torch_models_uri.setter
    def mlflow_torch_models_uri(self, value: dict[str, str]) -> None:
        self._loader.mlflow_torch_models_uri = value

    def _get_linked_models_uri(self) -> dict[str, Any]:
        return self._loader.get_all_uris()

    def _clear_linked_models_cache(self) -> None:
        self._loader.clear_cache()

    def _clear_ptmodel(self) -> None:
        self._loader._torch_model_instance = None

    # ------------------------------------------------------------------
    # Serialization — also clears loader cache on restore
    # ------------------------------------------------------------------

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._loader.clear_cache()


class _DatamintModelWrapper(BaseDatamintModel):
    def __init__(self, another_model: Any) -> None:
        super().__init__(settings=another_model.settings)
        self.another_model = another_model

    @property
    def task_type(self) -> TaskType | None:  # type: ignore[override]
        return getattr(self.another_model, 'task_type', None)

    @cached_property
    def _router(self) -> PredictionRouter:
        """The PredictionRouter instance responsible for dispatching predict calls."""
        return PredictionRouter(self.another_model, type(self.another_model))

    def load_context(self, context: PythonModelContext) -> None:
        self.another_model.load_context(context)


    def predict(self, model_input: list[Resource], params: dict[str, Any] | None = None) -> PredictionResult:
        return self.another_model.predict(model_input, params)
    
    def get_supported_modes(self) -> list[str]:
        return self.another_model.get_supported_modes()
    
    def predict_default(self, model_input: list[Resource], **kwargs: Any) -> PredictionResult:
        return self.another_model.predict_default(model_input, **kwargs)
    