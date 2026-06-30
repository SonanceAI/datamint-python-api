"""
Registry-driven prediction dispatcher and ``@prediction_mode`` decorator.

Replaces the hardcoded ``mode_param_keys`` dict, stub-method pattern,
and fragile ``_is_mode_implemented`` introspection previously in DatamintModel.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from .prediction_modes import PredictionMode

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModeSpec:
    """Metadata for a registered prediction mode handler."""

    mode: PredictionMode
    param_keys: tuple[str, ...] = ()
    fallback_to_default: bool = True


def bridge_mode(fn: Callable) -> Callable:
    """Marks a method as a bridge implementation.

    Bridge methods are only auto-registered in the router when their
    prerequisite mode is already in the registry (see
    ``PredictionRouter._BRIDGE_MODE_PREREQS``).  Without this marker the
    router would treat them as ordinary overrides and expose them regardless
    of whether the prerequisite is implemented.
    """
    fn._is_bridge = True  # type: ignore[attr-defined]
    return fn


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
    _DELEGATED_MODE_MAP = {
        PredictionMode.IMAGE: PredictionMode.SLICE,
    }
    # prerequisite → bridge modes that become available when prerequisite is registered
    _BRIDGE_MODE_PREREQS: dict[PredictionMode, PredictionMode] = {
        PredictionMode.VOLUME: PredictionMode.SLICE,
        PredictionMode.FRAME: PredictionMode.IMAGE,
        PredictionMode.ALL_FRAMES: PredictionMode.IMAGE,
    }

    def __init__(self, model_instance: Any,
                 base_class: type | None = None) -> None:
        self._registry = self.discover(model_instance, base_class)

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    @staticmethod
    def discover(model: Any, base_class: type | None) -> dict[PredictionMode, tuple[Callable, ModeSpec]]:
        """Build the mode -> handler registry from the model instance."""
        registry: dict[PredictionMode, tuple] = {}

        # Pass 1: decorator-based — walk the MRO __dict__ to avoid triggering
        # arbitrary property getters or descriptors on the instance.
        for cls in type(model).__mro__:
            for attr_name, raw in vars(cls).items():
                if attr_name.startswith("_"):
                    continue
                spec: ModeSpec | None = getattr(raw, "_mode_spec", None)
                if spec is not None and spec.mode not in registry:
                    registry[spec.mode] = (getattr(model, attr_name), spec)

        # Pass 2: convention-named (backward compat), skip if already registered
        for mode in PredictionMode:
            if mode in registry:
                continue
            method_name = f"predict_{mode.value}"
            method = getattr(model, method_name, None)
            if method is None:
                continue
            # Skip if the method is the unoverridden base-class stub
            if base_class is not None:
                base_method = getattr(base_class, method_name, None)
                if base_method is not None and getattr(method, "__func__", None) is base_method:
                    continue
            registry[mode] = (method, ModeSpec(mode=mode))

        # Step 3: add delegated modes based on _DELEGATED_MODE_MAP
        for target_mode, source_mode in PredictionRouter._DELEGATED_MODE_MAP.items():
            if target_mode in registry and source_mode not in registry:
                method_name = f"predict_{source_mode.value}"
                method = getattr(model, method_name, None)
                if method is not None:
                    _LOGGER.info(
                        "Registering handler for '%s' mode via delegation to '%s' handler",
                        source_mode.value, target_mode.value
                    )
                    registry[source_mode] = (method, ModeSpec(mode=source_mode))

        # Step 4: register bridge modes when their prerequisite is in the registry.
        # Bridge methods are marked with @bridge_mode; they are skipped in Pass 2
        # (as base-class methods) and only activated here when the prerequisite mode
        # is actually implemented.
        for bridge_mode_key, prereq_mode in PredictionRouter._BRIDGE_MODE_PREREQS.items():
            if bridge_mode_key in registry:
                continue
            if prereq_mode not in registry:
                continue
            method_name = f"predict_{bridge_mode_key.value}"
            method = getattr(model, method_name, None)
            if method is None:
                continue
            method_func = getattr(method, "__func__", method)
            if getattr(method_func, "_is_bridge", False):
                _LOGGER.info(
                    "Registering bridge handler for '%s' mode (prereq: '%s')",
                    bridge_mode_key.value, prereq_mode.value,
                )
                registry[bridge_mode_key] = (method, ModeSpec(mode=bridge_mode_key))

        return registry

    def update_registry(self, model_instance: Any,
                        base_class: type | None = None,
                        overwrite: bool = False) -> None:
        """Update the registry with handlers from a new model instance (e.g. linked model)."""
        new_entries = self.discover(model_instance, base_class)
        for mode, entry in new_entries.items():
            if mode in self._registry:
                if not overwrite:
                    _LOGGER.warning(
                        "Prediction handler for mode '%s' already exists. Use overwrite=True to replace it.",
                        mode.value,
                    )
                    continue
                _LOGGER.info(
                    "Updating prediction handler for mode '%s' with new handler from linked model instance.",
                    mode.value,
                )
            self._registry[mode] = entry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def supported_modes(self) -> list[str]:
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

    def _resolve_mode(self, model_input: list, params: dict) -> PredictionMode:
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

    def _get_handler(self, mode: PredictionMode) -> tuple[Callable, ModeSpec]:
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
