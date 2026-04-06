"""Combined LightningModule + BaseDatamintModel base for built-in trainers."""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

import lightning as L
from torch import Tensor
import torch

from datamint.mlflow.flavors.model import BaseDatamintModel, ModelSettings
from mlflow.pyfunc.model import PythonModelContext
import time
from mlflow.entities import Metric
from mlflow.tracking import MlflowClient

if TYPE_CHECKING:
    from datamint.mlflow.data import DatamintMLflowDataset

_LOGGER = logging.getLogger(__name__)
_SAMPLE_META_KEYS = {'resource_id', 'slice_index'}
SAMPLE_MAPPING_FILE = "test_sample_mapping.json"


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
        self._deferred_sample_batches: list[dict[str, Any]] = []
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
        """Collect per-sample metrics into a deferred buffer.

        Called from ``_common_step`` when ``_log_sample_metrics`` is enabled.
        Async CPU transfers are initiated but **not** synchronized here;
        resolution happens in :meth:`_resolve_deferred_batches` with a
        single sync before the metrics are flushed to MLflow.
        """
        resources = batch.get('resource', [])
        confidences = self._compute_sample_confidence(logits)
        sample_metrics = self._compute_sample_metrics(logits, batch)

        # Transfer all tensors to CPU asynchronously — no sync per batch.
        loss_cpu = loss_unreduced.detach().to('cpu', non_blocking=True)
        confidences_cpu = {key: vals.detach().to('cpu', non_blocking=True) for key, vals in confidences.items()}
        sample_metrics_cpu = {key: vals.detach().to('cpu', non_blocking=True) for key, vals in sample_metrics.items()}

        # Extract resource metadata (CPU-only, no GPU sync needed)
        resource_meta: list[dict[str, Any]] = []
        for i in range(logits.shape[0]):
            meta: dict[str, Any] = {}
            if i < len(resources):
                res = resources[i]
                if hasattr(res, 'parent_resource'):
                    meta['resource_id'] = res.parent_resource.id
                    meta['slice_index'] = res.slice_index
                else:
                    meta['resource_id'] = res.id
            resource_meta.append(meta)

        self._deferred_sample_batches.append({
            'resource_meta': resource_meta,
            'loss': loss_cpu,
            'confidences': confidences_cpu,
            'sample_metrics': sample_metrics_cpu,
            'batch_size': logits.shape[0],
        })

    def _compute_sample_confidence(self, logits: Tensor) -> dict[str, Tensor]:
        """Return per-sample confidence scores. Subclasses must override."""
        return {}

    def _compute_sample_metrics(self, logits: Tensor, batch: dict) -> dict[str, Tensor]:
        """Return per-sample metric values. Subclasses must override."""
        return {}

    def _resolve_deferred_batches(self) -> None:
        """Resolve all deferred async-transfer batches into ``_sample_buffer``.

        Issues a single ``torch.cuda.synchronize()`` to ensure all
        ``non_blocking`` CPU transfers have completed, then indexes into
        the batch tensors to build per-sample entries.
        """
        if not self._deferred_sample_batches:
            return

        # One sync for all accumulated batches instead of one per batch.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for batch_data in self._deferred_sample_batches:
            resource_meta = batch_data['resource_meta']
            loss_cpu = batch_data['loss']
            confidences_cpu = batch_data['confidences']
            sample_metrics_cpu = batch_data['sample_metrics']

            for i in range(batch_data['batch_size']):
                entry = dict(resource_meta[i])
                entry['loss'] = loss_cpu[i]
                for key, vals in confidences_cpu.items():
                    entry[key] = vals[i]
                for key, vals in sample_metrics_cpu.items():
                    entry[key] = vals[i]
                self._sample_buffer.append(entry)

        self._deferred_sample_batches.clear()

    def _flush_sample_metrics_to_mlflow(self) -> None:
        """Write accumulated sample data to MLflow and clear the buffer."""
        self._resolve_deferred_batches()

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

        _LOGGER.info("Flushing %d per-sample metrics (of %d samples) to MLflow...",
                     len(all_metrics), len(self._sample_buffer))

        if all_metrics:
            client = MlflowClient()
            try:
                client.log_batch(run_id, metrics=all_metrics, synchronous=True)
            except Exception as e:
                _LOGGER.error(f"Failed to log sample metrics batch: {e}")

        # Log resource_id → step mapping table as artifact.
        mapping_data = {
            "step": [],
            "resource_id": [],
            "slice_index": [],
            'metadata': {
                "dataset_name": dataset_name,
                "dataset_digest": dataset_digest,
                'model_id': self.mlflow_model_id,
                'timestamp': timestamp_ms,
            }
        }
        for step, entry in enumerate(self._sample_buffer):
            mapping_data["step"].append(step)
            mapping_data["resource_id"].append(entry.get("resource_id"))
            mapping_data["slice_index"].append(entry.get("slice_index"))
        try:
            mlflow.log_dict(
                {'test': mapping_data},
                artifact_file=SAMPLE_MAPPING_FILE,
                run_id=run_id,
            )
        except Exception as e:
            _LOGGER.warning(f"Failed to log sample mapping table: {e}")

        _LOGGER.info("Flushed per-sample metrics of %d samples to MLflow.", len(self._sample_buffer))
        self._sample_buffer.clear()

    def on_test_start(self) -> None:
        self.enable_sample_logging(True)
        self._sample_buffer.clear()
        self._deferred_sample_batches.clear()

    def on_test_epoch_end(self) -> None:
        self._flush_sample_metrics_to_mlflow()
        self.enable_sample_logging(False)

    # ------------------------------------------------------------------
    # MLflow lifecycle
    # ------------------------------------------------------------------

    def load_context(self, context: PythonModelContext) -> None:
        """Move weights to the configured device and set eval mode on MLflow load."""
        super().load_context(context)
        self.to(self.inference_device)
        self.eval()
