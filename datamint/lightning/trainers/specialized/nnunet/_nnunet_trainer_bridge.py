from __future__ import annotations

import json
import logging
import mlflow
from pathlib import Path

import importlib.metadata as _importlib_metadata

_LOGGER = logging.getLogger(__name__)

# ── version guard ──────────────────────────────────────────────────────────────
def _parse_version(v: str) -> tuple[int, ...]:
    try:
        return tuple(int(x) for x in v.split('.')[:3])
    except ValueError:
        return (0,)


try:
    _nnunetv2_version = _importlib_metadata.version('nnunetv2')
except _importlib_metadata.PackageNotFoundError:
    _nnunetv2_version = '0.0.0'

_ver = _parse_version(_nnunetv2_version)
if not ((2, 4, 0) <= _ver < (3, 0, 0)):
    raise ImportError(
        f"nnunetv2>=2.4,<3.0 is required for Datamint integration. "
        f"Currently installed: {_nnunetv2_version}. "
        f'Run: pip install "nnunetv2>=2.4,<3.0"'
    )
# ──────────────────────────────────────────────────────────────────────────────

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer 


# Keys logged by nnUNet that are not useful as MLflow metrics.
_SKIP_METRIC_KEYS = frozenset({'epoch_start_timestamps', 'epoch_end_timestamps'})


class _MLflowLogger:
    """MLflow sink compatible with nnUNet's MetaLogger external logger interface.

    Appended to ``self.logger.loggers`` so that every metric nnUNet emits via
    ``self.logger.log()`` is forwarded to the active MLflow run.
    """

    def update_config(self, config: dict) -> None:
        safe = {k: v for k, v in config.items() if isinstance(v, (int, float, str, bool))}
        if safe:
            try:
                mlflow.log_params(safe)
            except Exception:
                pass

    def log(self, key: str, value, step: int) -> None:
        if key in _SKIP_METRIC_KEYS:
            return
        try:
            mlflow.log_metric(key, float(value), step=step)
        except (TypeError, ValueError):
            pass

    def log_summary(self, key: str, value) -> None:
        try:
            mlflow.log_metric(key, float(value))
        except (TypeError, ValueError):
            pass


class _DatamintNNUNetTrainer(nnUNetTrainer):
    """nnUNetTrainer subclass that mirrors per-epoch metrics into MLflow.

    This is an internal implementation detail — use :class:`NNUNetTrainer`
    (the public Datamint trainer in ``trainer.py``) instead of instantiating
    this class directly.

    Metrics flow via ``self.logger.loggers`` (MetaLogger's external logger
    list) rather than by overriding ``self.log()`` — nnUNet calls
    ``self.logger.log()``, not ``self.log()``.

    Additional overrides: ``save_checkpoint`` (record best path + upload
    artifact) and ``perform_actual_validation`` (log per-class Dice from
    ``summary.json``).
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int | str,
        dataset_json: dict,
        device=None,
    ) -> None:
        import torch
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        super().__init__(plans, configuration, fold, dataset_json, device)
        self._best_checkpoint_path: Path | None = None
        # Register the MLflow sink so all nnUNet metrics flow to MLflow.
        self.logger.loggers.append(_MLflowLogger())

    # ── helpers that wrap super() calls so tests can patch them ───────────────

    def _super_save_checkpoint(self, filename: str, **kwargs) -> None:
        super().save_checkpoint(filename, **kwargs)

    # ── overrides ─────────────────────────────────────────────────────────────

    def save_checkpoint(self, filename: str, **kwargs) -> None:
        """Save checkpoint via nnUNet, record path, and upload to MLflow artifacts."""
        self._super_save_checkpoint(filename, **kwargs)
        self._best_checkpoint_path = Path(filename)
        mlflow.log_artifact(str(filename), artifact_path='nnunet_checkpoints')

    def _log_validation_summary(self) -> None:
        """Read nnUNet's summary.json and push per-class Dice scores to MLflow.

        Called at the end of each validation epoch by :meth:`perform_actual_validation`.
        Reads ``{output_folder}/validation/summary.json``.  In nnUNet v2,
        ``self.output_folder`` already includes ``fold_{fold}/``, so no extra
        nesting is needed.
        """
        summary_path = Path(self.output_folder) / 'validation' / 'summary.json'
        try:
            summary = json.loads(summary_path.read_text())
        except FileNotFoundError:
            _LOGGER.warning(
                "summary.json not found at '%s' — skipping per-class Dice logging.",
                summary_path,
            )
            return

        for class_name, metrics in summary.get('mean', {}).items():
            dice = metrics.get('Dice')
            if dice is not None:
                mlflow.log_metric(
                    f'val/dice_{class_name}', float(dice), step=self.current_epoch
                )

        foreground_mean = summary.get('foreground_mean')
        if foreground_mean is not None:
            mlflow.log_metric('val/dice_mean', float(foreground_mean), step=self.current_epoch)

    def perform_actual_validation(self, save_probabilities: bool = False) -> None:
        """Run nnUNet validation, then log per-class Dice to MLflow."""
        super().perform_actual_validation(save_probabilities)
        self._log_validation_summary()

    def print_to_log_file(self, *args, **kwargs) -> None:
        """Write to nnUNet's log file and also forward to Python logging."""
        super().print_to_log_file(*args, **kwargs)
        _LOGGER.info(' '.join(str(a) for a in args))
