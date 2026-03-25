"""Base trainer abstraction for Datamint training workflows.

Defines the :class:`BaseTrainer` template that orchestrates the full
pipeline: dataset → datamodule → model → Lightning Trainer → MLflow → deploy.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, TYPE_CHECKING
from functools import cached_property

import lightning as L
from torch import nn

from datamint.dataset.base import DatamintBaseDataset
from datamint.lightning.datamodule import DatamintDataModule

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.entities import Project

_LOGGER = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """Abstract base trainer encapsulating an end-to-end training workflow.

    Subclasses provide task-specific defaults for model architecture,
    transforms, loss, and metrics by overriding the ``_build_*`` /
    ``_default_*`` hooks.  Users typically only need to specify a
    ``project`` (or ``dataset``) and optionally override a few settings.

    Args:
        dataset: A pre-built :class:`DatamintBaseDataset`.  Mutually
            exclusive with *project*.
        project: Project name or :class:`Project` object used to
            auto-build a dataset when *dataset* is ``None``.
        model: A user-provided :class:`~lightning.LightningModule`.
            When ``None`` the trainer builds a default one via
            :meth:`_build_model`.
        loss_fn: Custom loss function forwarded to the default model.
            Ignored when *model* is provided (the user's module owns
            its own loss).
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        train_transform: Albumentations transform for training.  When
            ``None`` the trainer uses :meth:`_train_transform`.
        eval_transform: Albumentations transform for val/test.  When
            ``None`` the trainer uses :meth:`_eval_transform`.
        image_size: Target image size ``(H, W)`` or a single int for
            square images.  Forwarded to default transforms.  When
            ``None`` a sensible default is chosen.
        max_epochs: Maximum number of training epochs.
        early_stopping_patience: Epochs without improvement before
            stopping.  Set to ``None`` to disable early stopping.
        mlflow_experiment_name: MLflow experiment name.  Auto-generated
            from the project name when ``None``.
        register_model_name: Name for MLflow Model Registry.
            Auto-generated when ``None``.
        auto_deploy_adapter: When ``True``, auto-generate a
            :class:`~datamint.mlflow.flavors.model.DatamintModel`
            adapter after training.
        trainer_kwargs: Extra keyword arguments forwarded to
            :class:`lightning.Trainer`.
    """

    def __init__(
        self,
        dataset: DatamintBaseDataset | None = None,
        project: 'str | Project | None' = None,
        *,
        model: L.LightningModule | None = None,
        loss_fn: nn.Module | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        train_transform: 'BaseCompose | None' = None,
        eval_transform: 'BaseCompose | None' = None,
        image_size: int | tuple[int, int] | None = None,
        max_epochs: int = 50,
        early_stopping_patience: int | None = 10,
        mlflow_experiment_name: str | None = None,
        register_model_name: str | None = None,
        auto_deploy_adapter: bool = True,
        trainer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if dataset is None and project is None:
            raise ValueError("Either 'dataset' or 'project' must be provided.")
        if dataset is not None and project is not None:
            raise ValueError("'dataset' and 'project' are mutually exclusive.")

        self._user_dataset = dataset
        self._user_project = project
        self._user_model = model
        self._loss_fn = loss_fn
        self.batch_size = batch_size
        self.num_workers = num_workers
        self._user_train_transform = train_transform
        self._user_eval_transform = eval_transform
        if image_size is None:
            self.image_size: tuple[int, int] = (256, 256)
        elif isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.mlflow_experiment_name = mlflow_experiment_name
        self.register_model_name = register_model_name
        self.auto_deploy_adapter = auto_deploy_adapter
        self.trainer_kwargs = trainer_kwargs or {}

        # Populated during fit()
        self._datamodule: DatamintDataModule | None = None
        self._lightning_trainer: L.Trainer | None = None

    @cached_property
    def dataset(self) -> DatamintBaseDataset:
        return self._resolve_dataset()

    # ── Public API ──────────────────────────────────────────────
    def fit(self) -> dict[str, Any]:
        """Run the full training pipeline.

        Returns:
            Dictionary with keys ``'trainer'``, ``'model'``,
            ``'test_results'``, and ``'adapter'`` (when
            *auto_deploy_adapter* is enabled).
        """
        # clear cached properties in case of multiple calls to fit()
        for attr in ('dataset', 'model'):
            if attr in self.__dict__:
                del self.__dict__[attr]
        # 1. Resolve dataset (triggers @cached_property)
        _ = self.dataset

        # 2. Build transforms
        train_tf = self._user_train_transform or self._train_transform()
        eval_tf = self._user_eval_transform or self._eval_transform()

        # 3. Build DataModule
        self._datamodule = self._build_datamodule(self.dataset, train_tf, eval_tf)

        # 4. Build model
        _ = self.model  # triggers @cached_property to build the model

        # 5. Build callbacks & logger
        callbacks = self._build_callbacks()
        logger = self._build_logger()

        # 6. Build Lightning Trainer
        self._lightning_trainer = L.Trainer(
            max_epochs=self.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator='auto',
            **self.trainer_kwargs,
        )

        # 7. Train
        self._lightning_trainer.fit(self.model, datamodule=self._datamodule)

        # 8. Test
        test_results = self._lightning_trainer.test(datamodule=self._datamodule)

        # 9. Deploy adapter (only needed when the model is not already a DatamintModel)
        from datamint.mlflow.flavors.model import BaseDatamintModel
        adapter = None
        if self.auto_deploy_adapter and not isinstance(self.model, BaseDatamintModel):
            adapter = self._build_deploy_adapter()

        return {
            'trainer': self._lightning_trainer,
            'model': self.model,
            'test_results': test_results,
            'adapter': adapter,
        }

    # ── Template hooks (subclasses override these) ──────────────

    @abstractmethod
    def _build_dataset(self, project: 'str | Project') -> DatamintBaseDataset:
        """Build the appropriate dataset for this task."""
        ...

    @cached_property
    def model(self) -> L.LightningModule:
        if self._user_model is not None:
            return self._user_model
        else:
            loss = self._loss_fn or self._loss()
            metrics = self._metrics()
            return self._build_model(loss, metrics)

    @abstractmethod
    def _build_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Callable],
    ) -> L.LightningModule:
        """Build the default LightningModule for this task."""
        ...

    @abstractmethod
    def _train_transform(self) -> 'BaseCompose':
        """Return the training augmentation pipeline."""
        ...

    @abstractmethod
    def _eval_transform(self) -> 'BaseCompose':
        """Return the eval/test transform pipeline."""
        ...

    @abstractmethod
    def _loss(self) -> nn.Module:
        """Return the loss function for this task."""
        ...

    @abstractmethod
    def _metrics(self) -> dict[str, Callable]:
        """Return metrics as ``{name: factory_callable}``."""
        ...

    @abstractmethod
    def _monitor_metric(self) -> tuple[str, str]:
        """Return ``(metric_name, mode)`` for checkpointing / early stopping."""
        ...

    def _build_deploy_adapter(self) -> Any:
        """Build a DatamintModel deployment adapter.  Override in subclasses."""
        return None

    # ── Concrete helpers ────────────────────────────────────────

    def _resolve_dataset(self) -> DatamintBaseDataset:
        if self._user_dataset is not None:
            return self._user_dataset
        assert self._user_project is not None  # guaranteed by __init__ validation
        return self._build_dataset(self._user_project)

    def _build_datamodule(
        self,
        dataset: DatamintBaseDataset,
        train_transform: 'BaseCompose',
        eval_transform: 'BaseCompose',
    ) -> DatamintDataModule:
        return DatamintDataModule(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )

    def _build_callbacks(self) -> list:
        from datamint.mlflow.lightning.callbacks import MLFlowPyTorchModelCheckpoint, MLFlowDatamintModelCheckpoint
        from lightning.pytorch.callbacks import EarlyStopping
        from mlflow.pyfunc.model import PythonModel

        metric_name, mode = self._monitor_metric()
        project_name = self.dataset.project.name if self.dataset.project else 'datamint'
        model_name = self.register_model_name or project_name

        if isinstance(self.model, PythonModel):
            checkpoint_cls = MLFlowDatamintModelCheckpoint
        else:
            checkpoint_cls = MLFlowPyTorchModelCheckpoint

        callbacks: list = [
            checkpoint_cls(
                monitor=metric_name,
                mode=mode,
                save_top_k=1,
                register_model_name=model_name,
                register_model_on='test',
            )]

        if self.early_stopping_patience is not None:
            callbacks.append(EarlyStopping(
                monitor=metric_name,
                mode=mode,
                patience=self.early_stopping_patience,
            ))

        return callbacks

    def _build_logger(self):
        from lightning.pytorch.loggers import MLFlowLogger
        from datamint.mlflow import set_project

        project_name = self.dataset.project.name if self.dataset.project else 'datamint'
        set_project(project_name)

        experiment_name = self.mlflow_experiment_name or f"{project_name}_training"
        return MLFlowLogger(experiment_name=experiment_name)
