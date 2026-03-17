"""
DataMint Model Adapter Module

This module provides a flexible framework for wrapping ML models to work with DataMint's
annotation system. It supports various prediction modes for different data types and use cases.
"""

from typing import Any, TypeAlias
from abc import ABC
from enum import Enum
from dataclasses import dataclass
from mlflow.pyfunc import PyFuncModel, PythonModel, PythonModelContext
from datamint.entities.annotations import Annotation
from datamint.entities.resource import Resource
from datamint.mlflow.flavors.model_loader import LinkedModelLoader
from datamint.mlflow.flavors.prediction_router import PredictionRouter
import logging

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
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        invalid_fields = set(data.keys()) - valid_fields
        if invalid_fields:
            raise ValueError(f"Invalid fields for ModelSettings: {', '.join(sorted(invalid_fields))}")
        return cls(**data)


class PredictionMode(str, Enum):
    """
    Enumeration of supported prediction modes.

    Each mode corresponds to a specific method signature in DatamintModel.
    """
    # Standard modes
    DEFAULT = 'default'                # Default: process entire resource as-is

    # Simple modes
    IMAGE = 'image'                    # Process single 2d image resource

    # Video/temporal modes
    FRAME = 'frame'                    # Extract and process specific frame
    FRAME_RANGE = 'frame_range'        # Process contiguous frame range
    ALL_FRAMES = 'all_frames'          # Process all frames independently
    TEMPORAL_SEQUENCE = 'temporal_sequence'  # Process with temporal context window

    # 3D volume modes
    SLICE = 'slice'                    # Extract and process specific slice
    SLICE_RANGE = 'slice_range'        # Process contiguous slice range
    PRIMARY_SLICE = 'primary_slice'    # Process center/primary slice
    # MULTI_PLANE = 'multi_plane'        # Process multiple anatomical planes
    VOLUME = 'volume'                  # Process entire 3D volume

    # Spatial modes
    # ROI = 'roi'                        # Process single region of interest
    # MULTI_ROI = 'multi_roi'            # Process multiple regions
    # TILE = 'tile'                      # Split into tiles (whole slide imaging)
    # PATCH = 'patch'                    # Extract patches around points

    # Advanced modes
    INTERACTIVE = 'interactive'        # With user prompts (SAM-like)
    FEW_SHOT = 'few_shot'             # With context examples
    # MULTI_VIEW = 'multi_view'          # Multiple views of same subject


class DatamintModel(ABC, PythonModel):
    """Abstract adapter for wrapping ML models to produce Datamint annotations.

    Delegates model lifecycle to :class:`LinkedModelLoader` and prediction
    dispatch to :class:`PredictionRouter`.  Subclasses only need to override
    ``predict_default`` (and optionally other ``predict_*`` hooks).

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
    """

    # Keep for backward compat with subclass references
    LINKED_MODELS_DIR = "linked_models"

    def __init__(
        self,
        settings: ModelSettings | dict[str, Any] | None = None,
        mlflow_torch_models_uri: dict[str, str] | None = None,
        mlflow_models_uri: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._loader = LinkedModelLoader(
            mlflow_models_uri=mlflow_models_uri,
            mlflow_torch_models_uri=mlflow_torch_models_uri,
        )
        if isinstance(settings, dict):
            self.settings = ModelSettings.from_dict(settings)
        elif isinstance(settings, ModelSettings):
            self.settings = settings
        else:
            self.settings = ModelSettings()
        self._router: PredictionRouter | None = None

    # ------------------------------------------------------------------
    # Lifecycle (delegates to loader)
    # ------------------------------------------------------------------

    def load_context(self, context: PythonModelContext) -> None:
        """Called by MLflow when loading the model."""
        self._loader.load_all(context)

    @property
    def inference_device(self) -> str:
        return self._loader.inference_device

    def get_mlflow_models(self) -> dict[str, PyFuncModel]:
        """Access loaded MLflow models."""
        return self._loader.mlflow_models

    def get_mlflow_torch_models(self) -> dict[str, Any]:
        """Access loaded MLflow PyTorch models."""
        return self._loader.torch_models

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

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state.pop("_router", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._router = None
        self._loader.clear_cache()

    # ------------------------------------------------------------------
    # Prediction (delegates to router)
    # ------------------------------------------------------------------

    def predict(
        self,
        model_input: list[Resource],
        params: dict[str, Any] | None = None,
    ) -> PredictionResult:
        """Main prediction entry point.

        Routes to the appropriate handler based on ``params['mode']``.
        **Do not override** — implement ``predict_default`` (or other
        ``predict_*`` hooks) instead.
        """
        if self._router is None:
            self._router = PredictionRouter(self, DatamintModel)
        return self._router.dispatch(model_input, params or {})

    def get_supported_modes(self) -> list[str]:
        """Get list of prediction modes supported by this model."""
        if self._router is None:
            self._router = PredictionRouter(self, DatamintModel)
        return self._router.supported_modes()

    # ------------------------------------------------------------------
    # The only overridable prediction hook in the base
    # ------------------------------------------------------------------

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


