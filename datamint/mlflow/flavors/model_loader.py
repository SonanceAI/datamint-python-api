"""
Extracted model lifecycle management for DatamintModel.

Owns URI resolution, lazy model loading, cache lifecycle,
and serialization — all previously interleaved in the monolithic DatamintModel.
"""

import logging
import os
from collections.abc import Callable
from typing import Any

from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc import load_model as pyfunc_load_model
from mlflow.pytorch import load_model as pytorch_load_model
import torch

_LOGGER = logging.getLogger(__name__)

LINKED_MODELS_DIR = "linked_models"
_CACHED_ATTRS = frozenset({"_mlflow_models", "_mlflow_torch_models", "_device"})


class LinkedModelLoader:
    """Owns URI resolution, lazy model loading, and cache lifecycle.

    Extracted from DatamintModel so prediction routing and model lifecycle are independent.
    Device detection is handled by :class:`~datamint.mlflow.flavors.model.BaseDatamintModel`.
    """

    def __init__(
        self,
        mlflow_models_uri: dict[str, str] | None = None,
        mlflow_torch_models_uri: dict[str, str] | None = None,
        torch_model: torch.nn.Module | None = None,
    ) -> None:
        self.mlflow_models_uri: dict[str, str] = (mlflow_models_uri or {}).copy()
        self.mlflow_torch_models_uri: dict[str, str] = (mlflow_torch_models_uri or {}).copy()
        self._torch_model_instance: torch.nn.Module | None = torch_model

    # --- Loading ----------------------------------------------------------

    def load_all(self, device: str) -> None:
        self._device = device
        self._mlflow_models = self._load_pyfunc_models(device)
        self._mlflow_torch_models = self._load_torch_models(device)
        if self._torch_model_instance and hasattr(self._torch_model_instance, "eval"):
            self._torch_model_instance.eval()

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

    def _load_pyfunc_models(self, device: str) -> dict[str, PyFuncModel]:
        return self._load_generic(
            self.mlflow_models_uri,
            pyfunc_load_model,
            model_config={"device": device},
        )

    def _load_torch_models(self, device: str) -> dict[str, Any]:
        models = self._load_generic(
            self.mlflow_torch_models_uri,
            pytorch_load_model,
            device=device,
            map_location=device,
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
            self._mlflow_models = self._load_pyfunc_models(getattr(self, "_device", "cpu"))
        return self._mlflow_models

    @property
    def torch_models(self) -> dict[str, Any]:
        if not hasattr(self, "_mlflow_torch_models"):
            _LOGGER.warning("Loading MLflow PyTorch models on first access")
            self._mlflow_torch_models = self._load_torch_models(getattr(self, "_device", "cpu"))
        ret = self._mlflow_torch_models.copy()
        if self._torch_model_instance:
            ret["default"] = self._torch_model_instance
        return ret

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
