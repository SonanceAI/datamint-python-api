"""Benchmark: compare several trainers against one shared dataset.

Not a new training system, each trainer runs its own complete
:meth:`~datamint.lightning.trainers.base_trainer.BaseTrainer.fit` pipeline.
``Benchmark`` guarantees every trainer sees the same dataset and split, and
collects their results into a single ranked leaderboard.
"""

from __future__ import annotations

import importlib
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml

from datamint.lightning.trainers.base_trainer import BaseTrainer
from datamint.lightning.trainers.classification_trainer import ClassificationTrainer
from datamint.lightning.trainers.detection_trainer import DetectionTrainer
from datamint.lightning.trainers.seg2d_trainer import SemanticSegmentation2DTrainer
from datamint.lightning.trainers.vol_seg_trainer import VolumeSegmentationTrainer

if TYPE_CHECKING:
    from datamint.dataset.base import DatamintBaseDataset

_LOGGER = logging.getLogger(__name__)

TrainerSpec = tuple[type[BaseTrainer], dict[str, Any]]

# Common task-family ancestors. A trainer that doesn't descend from any of
# these (e.g. NNUNetTrainer, which subclasses BaseTrainer directly) falls
# back to being its own family -- it can only be benchmarked against other
# instances of that same class.
_TASK_FAMILIES: tuple[type[BaseTrainer], ...] = (
    ClassificationTrainer,
    SemanticSegmentation2DTrainer,
    VolumeSegmentationTrainer,
    DetectionTrainer,
)


def _task_family(trainer_cls: type[BaseTrainer]) -> type[BaseTrainer]:
    """Return the task-family root for *trainer_cls*, or the class itself if none match."""
    for family in _TASK_FAMILIES:
        if issubclass(trainer_cls, family):
            return family
    return trainer_cls


def _to_scalar(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, 'item'):
        return value.item()
    return value


def _utc_now_isoformat() -> str:
    return datetime.now(timezone.utc).isoformat()


def _class_dotted_path(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _resolve_trainer_class(dotted_path: str) -> type[BaseTrainer]:
    module_path, _, class_name = dotted_path.rpartition('.')
    if not module_path:
        raise ValueError(f"Invalid trainer class path {dotted_path!r}; expected 'module.ClassName'.")
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if not (isinstance(cls, type) and issubclass(cls, BaseTrainer)):
        raise TypeError(f"{dotted_path!r} does not resolve to a BaseTrainer subclass.")
    return cls


class Benchmark:
    """Train several trainers sequentially against one shared dataset/split and rank them.

    Every trainer is constructed with the same dataset and the
    same ``split_as_of_timestamp``, so they all resolve the identical
    project-scoped split assignments. This requires the dataset's project to
    already have split assignments.

    All trainers must share one task-family ancestor (classification,
    2-D segmentation, volume segmentation, or detection). 
    A trainer that doesn't descend from any of those (e.g. ``NNUNetTrainer``) 
    is treated as its own family and can only be
    benchmarked against other instances of that same class.

    Args:
        dataset: Pre-built dataset shared across every trainer.
        trainers: ``(TrainerClass, kwargs)`` pairs. Each ``kwargs`` dict must
            include a unique, explicit ``model_name``.
        split_as_of_timestamp: Timestamp pinning the project-scoped split
            assignments every trainer resolves against. When omitted,
            captured once (UTC now) at construction.
        main_metric: Bare metric key (e.g. ``'accuracy'``, ``'dice'``) shared
            by every trainer's ``val/`` and ``test/`` logged metric names.
            Used as the leaderboard's sort column. When omitted, falls back
            to the first trainer's ``_monitor_metric()``.
        main_metric_mode: ``'max'`` (default) or ``'min'``. Sort direction
            for *main_metric*. Ignored when main_metric is omitted (the
            mode from ``_monitor_metric()`` is used instead).
        **shared_kwargs: Extra kwargs forwarded to every trainer's
            constructor (e.g. ``max_epochs=``). Per-trainer kwargs win on
            conflict.

    Example::

        from datamint import ImageDataset
        from datamint.lightning import Benchmark
        from datamint.lightning.trainers import UNetPPTrainer, DeepLabV3PlusTrainer

        dataset = ImageDataset(project='BUS_Segmentation', return_segmentations=True)

        bench = Benchmark(
            dataset=dataset,
            trainers=[
                (UNetPPTrainer, {'encoder_name': 'resnet34', 'model_name': 'unetpp_r34'}),
                (DeepLabV3PlusTrainer, {'model_name': 'deeplabv3plus'}),
            ],
            main_metric='dice',
        )
        leaderboard = bench.run()

        bench.save_config('benchmark.yaml')
        # later: Benchmark.load_from_file('benchmark.yaml', dataset=dataset).run()
    """

    def __init__(
        self,
        dataset: DatamintBaseDataset,
        trainers: list[TrainerSpec],
        split_as_of_timestamp: str | None = None,
        main_metric: str | None = None,
        main_metric_mode: str = 'max',
        **shared_kwargs: Any,
    ) -> None:
        if not trainers:
            raise ValueError("'trainers' must contain at least one (TrainerClass, kwargs) pair.")
        if main_metric_mode not in ('max', 'min'):
            raise ValueError(f"main_metric_mode must be 'max' or 'min', got {main_metric_mode!r}.")

        self.dataset = dataset
        self._specs: list[TrainerSpec] = list(trainers)
        self.split_as_of_timestamp = split_as_of_timestamp or _utc_now_isoformat()
        self.main_metric = main_metric
        self.main_metric_mode = main_metric_mode
        self._shared_kwargs = shared_kwargs

        self._validate_same_task_family()
        self._validate_unique_model_names()

    def _validate_same_task_family(self) -> None:
        families = {_task_family(cls) for cls, _ in self._specs}
        if len(families) > 1:
            names = sorted(f.__name__ for f in families)
            raise ValueError(
                f"Cannot benchmark trainers from different task families: {names}. "
                "All trainers in a Benchmark must share a common task-family ancestor."
            )

    def _validate_unique_model_names(self) -> None:
        names = [kwargs.get('model_name') for _, kwargs in self._specs]
        if any(name is None for name in names):
            raise ValueError("Every trainer spec needs an explicit 'model_name' for benchmarking.")
        if len(set(names)) != len(names):
            dupes = sorted({name for name in names if names.count(name) > 1})
            raise ValueError(
                f"Duplicate model_name(s) across trainer specs: {dupes}. "
                "Each trainer needs a unique model_name."
            )

    @staticmethod
    def _find_checkpoint_callback(lightning_trainer: Any) -> Any | None:
        for callback in lightning_trainer.callbacks:
            if hasattr(callback, 'registered_model_info'):
                return callback
        return None

    def _metric_name_and_mode(self, trainer: BaseTrainer) -> tuple[str, str]:
        if self.main_metric is not None:
            return f"val/{self.main_metric}", self.main_metric_mode
        return trainer._monitor_metric()

    def run(self) -> pd.DataFrame:
        """Train every trainer sequentially and return a leaderboard DataFrame."""
        rows: list[dict[str, Any]] = []
        test_col = val_col = None
        mode = self.main_metric_mode

        for trainer_cls, kwargs in self._specs:
            model_name = kwargs['model_name']
            _LOGGER.info("Benchmark: training '%s' (%s)...", model_name, trainer_cls.__name__)

            trainer = trainer_cls(
                dataset=self.dataset,
                split_as_of_timestamp=self.split_as_of_timestamp,
                **{**self._shared_kwargs, **kwargs},
            )
            result = trainer.fit()

            if result.get('paused'):
                raise RuntimeError(
                    f"Training for '{model_name}' was interrupted before completion "
                    f"(run_id={result['run_id']}); Benchmark does not support resuming mid-run."
                )

            val_col, mode = self._metric_name_and_mode(trainer)
            test_col = f"test/{val_col.removeprefix('val/')}"

            test_results = result['test_results'][0] if result['test_results'] else {}
            checkpoint_cb = self._find_checkpoint_callback(trainer._lightning_trainer)
            registered_info = getattr(checkpoint_cb, 'registered_model_info', None)

            rows.append({
                'model_name': model_name,
                'trainer_class': trainer_cls.__name__,
                val_col: _to_scalar(getattr(checkpoint_cb, 'best_model_score', None)),
                test_col: test_results.get(test_col),
                'run_id': trainer._lightning_trainer.logger.run_id,
                'registered_version': registered_info.version if registered_info is not None else None,
            })

        leaderboard = pd.DataFrame(rows)
        sort_col = test_col if leaderboard[test_col].notna().any() else val_col
        return leaderboard.sort_values(sort_col, ascending=(mode == 'min')).reset_index(drop=True)

    def save_config(self, path: str) -> None:
        """Save this benchmark's trainer specs and settings as a YAML file. """
        
        config = {
            'split_as_of_timestamp': self.split_as_of_timestamp,
            'main_metric': self.main_metric,
            'main_metric_mode': self.main_metric_mode,
            'shared_kwargs': self._shared_kwargs,
            'trainers': [
                {'trainer_class': _class_dotted_path(cls), 'kwargs': kwargs}
                for cls, kwargs in self._specs
            ],
        }
        with open(path, 'w') as f:
            yaml.safe_dump(config, f, sort_keys=False)

    @classmethod
    def load_from_file(cls, path: str, dataset: DatamintBaseDataset) -> Benchmark:
        """Reconstruct a :class:`Benchmark` from a file saved by :meth:`save_config`.

        Args:
            path: Path to the YAML file.
            dataset: Freshly-built dataset to benchmark against.
        """
        with open(path) as f:
            config = yaml.safe_load(f)

        trainers = [
            (_resolve_trainer_class(spec['trainer_class']), spec['kwargs'])
            for spec in config['trainers']
        ]
        return cls(
            dataset=dataset,
            trainers=trainers,
            split_as_of_timestamp=config.get('split_as_of_timestamp'),
            main_metric=config.get('main_metric'),
            main_metric_mode=config.get('main_metric_mode', 'max'),
            **config.get('shared_kwargs', {}),
        )
