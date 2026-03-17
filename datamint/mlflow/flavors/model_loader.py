"""
Extracted model lifecycle management for DatamintModel.

Owns URI resolution, lazy model loading, device detection, cache lifecycle,
and serialization — all previously interleaved in the monolithic DatamintModel.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from typing import Any

from mlflow.environment_variables import MLFLOW_DEFAULT_PREDICTION_DEVICE
from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc import load_model as pyfunc_load_model
from mlflow.pyfunc.model import PythonModelContext
from mlflow.pytorch import load_model as pytorch_load_model

_LOGGER = logging.getLogger(__name__)

LINKED_MODELS_DIR = "linked_models"
_CACHED_ATTRS = frozenset({"_mlflow_models", "_mlflow_torch_models", "_inference_device"})


class LinkedModelLoader:
    """Owns URI resolution, lazy model loading, device management, and cache lifecycle.

    Extracted from DatamintModel so prediction routing and model lifecycle are independent.
    """

    def __init__(
        self,
        mlflow_models_uri: dict[str, str] | None = None,
        mlflow_torch_models_uri: dict[str, str] | None = None,
    ) -> None:
        self.mlflow_models_uri: dict[str, str] = (mlflow_models_uri or {}).copy()
        self.mlflow_torch_models_uri: dict[str, str] = (mlflow_torch_models_uri or {}).copy()

    # --- Device -----------------------------------------------------------

    @property
    def inference_device(self) -> str:
        if hasattr(self, "_inference_device") and self._inference_device is not None:
            return self._inference_device
        env_device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
        if env_device:
            _LOGGER.info("Inference device not set; getting from environment variable (%s)", env_device)
            return env_device
        _LOGGER.warning("Inference device not set; defaulting to 'cpu'")
        return "cpu"

    def detect_device(self, context: PythonModelContext | None = None) -> str:
        import torch

        device = None
        if context and context.model_config:
            device = context.model_config.get("device", None)
            _LOGGER.info("Model config device: %s", device)
        if device is None:
            env_device = MLFLOW_DEFAULT_PREDICTION_DEVICE.get()
            if env_device:
                device = env_device
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        _LOGGER.info("Set inference device: %s", device)
        self._inference_device = device
        return device

    # --- Loading ----------------------------------------------------------

    def load_all(self, context: PythonModelContext | None = None) -> None:
        self.detect_device(context)
        self._mlflow_models = self._load_pyfunc_models()
        self._mlflow_torch_models = self._load_torch_models()

    def _resolve_uri(self, uri: str) -> str:
        if os.path.exists(uri):
            return os.path.abspath(uri)
        if uri.startswith("models:/"):
            local = uri.replace("models:/", f"{LINKED_MODELS_DIR}/", 1)
            if os.path.exists(local):
                _LOGGER.info("Model found locally at '%s'", local)
                return os.path.abspath(local)
        return uri

    def _load_generic(
        self, uris: dict[str, str], loader: Callable, **kwargs: Any
    ) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, uri in uris.items():
            resolved = self._resolve_uri(uri)
            loaded[name] = loader(resolved, **kwargs)
            _LOGGER.info("Loaded model '%s' from %s", name, resolved)
        return loaded

    def _load_pyfunc_models(self) -> dict[str, PyFuncModel]:
        return self._load_generic(
            self.mlflow_models_uri,
            pyfunc_load_model,
            model_config={"device": self.inference_device},
        )

    def _load_torch_models(self) -> dict[str, Any]:
        models = self._load_generic(
            self.mlflow_torch_models_uri,
            pytorch_load_model,
            device=self.inference_device,
            map_location=self.inference_device,
        )
        for m in models.values():
            if hasattr(m, "eval"):
                m.eval()
        return models

    # --- Access (lazy) ----------------------------------------------------

    @property
    def mlflow_models(self) -> dict[str, PyFuncModel]:
        if not hasattr(self, "_mlflow_models"):
            _LOGGER.warning("Loading MLflow models on first access")
            self._mlflow_models = self._load_pyfunc_models()
        return self._mlflow_models

    @property
    def torch_models(self) -> dict[str, Any]:
        if not hasattr(self, "_mlflow_torch_models"):
            _LOGGER.warning("Loading MLflow PyTorch models on first access")
            self._mlflow_torch_models = self._load_torch_models()
        return self._mlflow_torch_models

    # --- Linked model URIs ------------------------------------------------

    def get_all_uris(self) -> dict[str, str]:
        return {**self.mlflow_models_uri, **self.mlflow_torch_models_uri}

    # --- Serialization ----------------------------------------------------

    def clear_cache(self) -> None:
        for attr in _CACHED_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for attr in _CACHED_ATTRS:
            state.pop(attr, None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.clear_cache()
