"""Abstract base trainer for object detection tasks."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

from torch import nn

from datamint.dataset.image_dataset import ImageDataset, detection_collate_fn
from datamint.lightning.datamodule import DatamintDataModule
from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from datamint.entities import Project


class DetectionTrainer(BaseTrainer):
    """Abstract trainer for object detection tasks.

    Provides shared defaults for all detection models:

    * **Dataset** – :class:`~datamint.dataset.ImageDataset` with ``return_boxes=True``
    * **Collate** – :func:`~datamint.dataset.detection_collate_fn` (variable-length boxes)
    * **Metrics** – Mean Average Precision (torchmetrics)
    * **Monitor** – ``val/map`` (maximise)

    Subclasses must implement :meth:`_build_model`, :meth:`_train_transform`,
    and :meth:`_eval_transform`.
    """

    def _build_dataset(
        self,
        project: 'str | Project',
        **kwargs: Any,
    ) -> ImageDataset:
        return ImageDataset(project=project, return_boxes=True, **kwargs)

    def _build_datamodule(
        self,
        dataset: ImageDataset,
        train_transform: Any,
        eval_transform: Any,
    ) -> DatamintDataModule:
        return DatamintDataModule(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            split_as_of_timestamp=self.split_as_of_timestamp,
            train_transform=train_transform,
            eval_transform=eval_transform,
            pin_memory=False,
            collate_fn=detection_collate_fn,
        )

    def _loss(self) -> None:  # type: ignore[override]
        # YOLOX (and most modern detection heads) compute loss internally.
        # Returning None signals _build_model that no external loss module is needed.
        return None

    def _metrics(self) -> dict[str, Callable]:
        # Detection modules (e.g. YOLOXModule) build and own their metrics
        # internally. No external factory is needed by BaseTrainer.
        return {}

    def _monitor_metric(self) -> tuple[str, str]:
        return 'val/map', 'max'
