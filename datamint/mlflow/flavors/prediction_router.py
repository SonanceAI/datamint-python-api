"""
Registry-driven prediction dispatcher and ``@prediction_mode`` decorator.

Replaces the hardcoded ``mode_param_keys`` dict, stub-method pattern,
and fragile ``_is_mode_implemented`` introspection previously in DatamintModel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datamint.mlflow.flavors.model import PredictionMode

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """Metadata for a registered prediction mode handler."""

    mode: PredictionMode
    param_keys: tuple[str, ...] = ()
    fallback_to_default: bool = True


def prediction_mode(
    mode: PredictionMode,
    *,
    param_keys: tuple[str, ...] = (),
    fallback_to_default: bool = True,
) -> Callable:
    """Decorator that registers a method as a prediction mode handler.

    Usage::

        class MyModel(DatamintModel):
            @prediction_mode(PredictionMode.SLICE, param_keys=("slice_index", "axis"))
            def predict_slice(self, model_input, *, slice_index, axis="axial", **kw):
                ...
    """

    def decorator(fn: Callable) -> Callable:
        fn._mode_spec = ModeSpec(  # type: ignore[attr-defined]
            mode=mode,
            param_keys=param_keys,
            fallback_to_default=fallback_to_default,
        )
        return fn

    return decorator


class PredictionRouter:
    """Registry-driven prediction dispatcher.

    Discovers mode handlers in two ways (in order of priority):

    1. Methods decorated with ``@prediction_mode``.
    2. Convention-named methods ``predict_<mode_value>`` **not** defined on the
       abstract base (backward compatibility with old-style overrides).
    """

    _RESERVED_PARAMS = frozenset({"mode", "confidence_threshold"})

    def __init__(self, model_instance: Any, base_class: type) -> None:
        from .model import PredictionMode  # local import to avoid circular deps

        self._PredictionMode = PredictionMode
        self._model = model_instance
        self._base_class = base_class
        self._registry: dict[PredictionMode, tuple[Callable, ModeSpec]] = {}
        self._discover()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover(self) -> None:
        """Build the mode -> handler registry from the model instance."""
        PredictionMode = self._PredictionMode

        # Pass 1: decorator-based
        for attr_name in dir(self._model):
            if attr_name.startswith("_"):
                continue
            method = getattr(self._model, attr_name, None)
            spec: ModeSpec | None = getattr(method, "_mode_spec", None)
            if spec is not None:
                self._registry[spec.mode] = (method, spec)

        # Pass 2: convention-named (backward compat), skip if already registered
        for mode in PredictionMode:
            if mode in self._registry:
                continue
            method_name = f"predict_{mode.value}"
            method = getattr(self._model, method_name, None)
            if method is None:
                continue
            # Skip if the method is the unoverridden base-class stub
            base_method = getattr(self._base_class, method_name, None)
            if base_method is not None and getattr(method, "__func__", None) is base_method:
                continue
            self._registry[mode] = (method, ModeSpec(mode=mode))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def supported_modes(self) -> list[str]:
        PredictionMode = self._PredictionMode
        return [mode.value for mode in PredictionMode if mode in self._registry]

    def dispatch(
        self,
        model_input: list,
        params: dict[str, Any],
    ) -> list:
        mode = self._resolve_mode(model_input, params)
        _LOGGER.info(
            "Received prediction request with %d resources and params %s with mode '%s'",
            len(model_input), params, mode.value,
        )
        handler, spec = self._get_handler(mode)
        _LOGGER.info("Routing to '%s' mode for %d resources", mode.value, len(model_input))
        mode_kw, common_kw = self._split_params(params, spec)
        result = handler(model_input, **mode_kw, **common_kw)
        return self._post_process(result, params)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_mode(self, model_input: list, params: dict) -> Any:
        PredictionMode = self._PredictionMode
        mode_str = params.get("mode", PredictionMode.DEFAULT.value)
        try:
            is_all_image = all(
                getattr(r, "mimetype", "").startswith("image/") for r in model_input
            )
        except Exception:
            is_all_image = False

        _LOGGER.debug("Parsing prediction mode: '%s' | is_all_image=%s", mode_str, is_all_image)

        if mode_str == PredictionMode.DEFAULT.value and is_all_image:
            mode_str = PredictionMode.IMAGE.value
        try:
            return PredictionMode(mode_str)
        except ValueError:
            valid = [m.value for m in PredictionMode]
            raise ValueError(
                f"Invalid prediction mode: '{mode_str}'\n"
                f"Valid modes: {', '.join(valid)}"
            )

    def _get_handler(self, mode: Any) -> tuple[Callable, ModeSpec]:
        PredictionMode = self._PredictionMode
        if mode in self._registry:
            return self._registry[mode]
        if PredictionMode.DEFAULT in self._registry:
            _LOGGER.info("Mode '%s' not implemented, falling back to default", mode.value)
            return self._registry[PredictionMode.DEFAULT]
        available = self.supported_modes()
        raise NotImplementedError(
            f"Prediction mode '{mode.value}' is not supported by this model.\n"
            f"Supported modes: {', '.join(available)}\n"
            f"Implement predict_{mode.value}() to add support for this mode."
        )

    def _split_params(
        self, params: dict[str, Any], spec: ModeSpec
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        mode_kw: dict[str, Any] = {}
        common_kw: dict[str, Any] = {}
        spec_keys = set(spec.param_keys)
        for k, v in params.items():
            if k in self._RESERVED_PARAMS:
                continue
            if k in spec_keys:
                mode_kw[k] = v
            else:
                common_kw[k] = v
        return mode_kw, common_kw

    @staticmethod
    def _post_process(result: list, params: dict[str, Any]) -> list:
        threshold = params.get("confidence_threshold")
        if threshold is not None:
            result = [
                [a for a in preds if getattr(a, "confiability", 1.0) >= threshold]
                for preds in result
            ]
            _LOGGER.debug("Applied confidence threshold: %s", threshold)
        return result
