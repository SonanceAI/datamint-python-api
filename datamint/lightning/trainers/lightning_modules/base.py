"""Combined LightningModule + BaseDatamintModel base for built-in trainers."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import lightning as L
from torch import Tensor

from datamint.mlflow.flavors.model import BaseDatamintModel, ModelSettings
from mlflow.pyfunc.model import PythonModelContext

if TYPE_CHECKING:
    from datamint.mlflow.data import DatamintMLflowDataset

_LOGGER = logging.getLogger(__name__)
_SAMPLE_META_KEYS = {'resource_id', 'slice_index'}


class DatamintLightningModule(L.LightningModule, BaseDatamintModel):
    """A :class:`~lightning.LightningModule` that is also a
    :class:`~datamint.mlflow.flavors.model.BaseDatamintModel`.

    Built-in trainers use this as the base for their default models so that
    the trained module can be logged once with ``datamint_flavor`` — no
    separate adapter step is required.
    """

    def __init__(self, settings: ModelSettings | None = None) -> None:
        L.LightningModule.__init__(self)
        BaseDatamintModel.__init__(self, settings=settings)
        self._log_sample_metrics: bool = False
        self._sample_buffer: list[dict[str, Any]] = []
        self.mlflow_model_id: str | None = None  # Injected by modelcheckpoint at runtime. FIXME: This is a bit hacky

    # ------------------------------------------------------------------
    # Per-sample metrics logging
    # ------------------------------------------------------------------

    def enable_sample_logging(self, enabled: bool = True) -> None:
        """Enable or disable per-sample metric accumulation."""
        self._log_sample_metrics = enabled

    def _accumulate_sample_data(
        self,
        batch: dict,
        logits: Tensor,
        loss_unreduced: Tensor,
        stage: str,
    ) -> None:
        """Collect per-sample metrics into ``_sample_buffer``.

        Called from ``_common_step`` when ``_log_sample_metrics`` is enabled.
        """
        resources = batch.get('resource', [])
        confidences = self._compute_sample_confidence(logits)
        sample_metrics = self._compute_sample_metrics(logits, batch)

        for i in range(logits.shape[0]):
            entry: dict[str, Any] = {}
            if i < len(resources):
                res = resources[i]
                if hasattr(res, 'parent_resource'):
                    entry['resource_id'] = res.parent_resource.id
                    entry['slice_index'] = res.slice_index
                else:
                    entry['resource_id'] = res.id
            entry['loss'] = loss_unreduced[i].item()
            for key, vals in confidences.items():
                entry[key] = vals[i].item()
            for key, vals in sample_metrics.items():
                entry[key] = vals[i].item()
            self._sample_buffer.append(entry)

    def _compute_sample_confidence(self, logits: Tensor) -> dict[str, Tensor]:
        """Return per-sample confidence scores. Subclasses must override."""
        return {}

    def _compute_sample_metrics(self, logits: Tensor, batch: dict) -> dict[str, Tensor]:
        """Return per-sample metric values. Subclasses must override."""
        return {}

    def _flush_sample_metrics_to_mlflow(self) -> None:
        """Write accumulated sample data to MLflow and clear the buffer."""
        if not self._sample_buffer:
            return

        import mlflow
        from datamint.mlflow.models import _get_MLFlowLogger

        logger = _get_MLFlowLogger(self.trainer)
        if logger is None or logger.run_id is None:
            _LOGGER.warning(
                "No MLFlowLogger with run_id found. "
                "Skipping per-sample metrics flush."
            )
            return

        run_id = logger.run_id
        mlflow_dataset: DatamintMLflowDataset | None = getattr(logger, '_mlflow_dataset', None)
        if mlflow_dataset is None:
            _LOGGER.info(
                "MLFlowLogger does not have '_mlflow_dataset' attribute. "
                "Per-sample metrics will be logged without dataset association."
            )
            dataset_name = None
            dataset_digest = None
        else:
            dataset_name = mlflow_dataset.name
            dataset_digest = mlflow_dataset.digest

        if self.mlflow_model_id is None:
            _LOGGER.warning(
                "MLflow model ID is not set on the LightningModule. "
                "Per-sample metrics will be logged without model association."
            )

        # Log per-sample metrics with step = sample index.
        # Build a flat list of Metric objects and send in one log_batch call
        # instead of one log_metrics call per sample (N → 1 HTTP round-trips).
        import time
        from mlflow.entities import Metric
        from mlflow.tracking import MlflowClient

        metric_keys = {
            key
            for entry in self._sample_buffer
            for key in entry
            if key not in _SAMPLE_META_KEYS
        }

        timestamp_ms = int(time.time() * 1000)
        all_metrics: list[Metric] = [
            Metric(key=f"test/sample/{key}", value=float(entry[key]),
                   timestamp=timestamp_ms, step=step,
                   dataset_name=dataset_name, dataset_digest=dataset_digest,
                   run_id=run_id,
                   model_id=self.mlflow_model_id)
            for step, entry in enumerate(self._sample_buffer)
            for key in metric_keys
            if entry.get(key) is not None
        ]

        _LOGGER.info("Flushing %d per-sample metrics (of %d samples) to MLflow...", len(all_metrics), len(self._sample_buffer))

        if all_metrics:
            client = MlflowClient()
            try:
                client.log_batch(run_id, metrics=all_metrics, synchronous=True)
            except Exception as e:
                _LOGGER.error(f"Failed to log sample metrics batch: {e}")

        # Log resource_id → step mapping table as artifact.
        mapping_data: dict[str, list[Any]] = {
            "step": [],
            "resource_id": [],
            "slice_index": [],
        }
        for step, entry in enumerate(self._sample_buffer):
            mapping_data["step"].append(step)
            mapping_data["resource_id"].append(entry.get("resource_id"))
            mapping_data["slice_index"].append(entry.get("slice_index"))
        try:
            mlflow.log_table(
                data=mapping_data,
                artifact_file="test_sample_mapping.json",
            )
        except Exception as e:
            _LOGGER.warning(f"Failed to log sample mapping table: {e}")

        _LOGGER.info("Flushed per-sample metrics of %d samples to MLflow.", len(self._sample_buffer))
        self._sample_buffer.clear()

    def on_test_start(self) -> None:
        self.enable_sample_logging(True)
        self._sample_buffer.clear()

    def on_test_epoch_end(self) -> None:
        self._flush_sample_metrics_to_mlflow()
        self.enable_sample_logging(False)

    # ------------------------------------------------------------------
    # MLflow lifecycle
    # ------------------------------------------------------------------

    def load_context(self, context: PythonModelContext) -> None:
        """Move weights to the configured device and set eval mode on MLflow load."""
        device = (context.model_config or {}).get('device', 'cpu')
        self.to(device)
        self.eval()
