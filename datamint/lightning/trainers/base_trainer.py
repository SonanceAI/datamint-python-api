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
import mlflow

from datamint.dataset.base import DatamintBaseDataset
from datamint.lightning.datamodule import DatamintDataModule
from datamint.mlflow import set_project
from datamint.mlflow.flavors.model import BaseDatamintModel

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
        self._lightning_trainer: L.Trainer | None = None

    @cached_property
    def dataset(self) -> DatamintBaseDataset:
        return self._resolve_dataset()

    @property
    def _project_name(self) -> str:
        if self._user_project is None:
            project_name = self.dataset.project.name if self.dataset.project else 'datamint'
        elif isinstance(self._user_project, str):
            project_name = self._user_project
        else:
            project_name = self._user_project.name

        return project_name

    @property
    def experiment_name(self) -> str:
        return self.mlflow_experiment_name or f"{self._project_name}_training"

    def _with_project(self):
        set_project(self._project_name)

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

        self._with_project()  # ensure MLflow project context is set for the entire training pipeline
        exp = mlflow.set_experiment(self.experiment_name)  # ensure experiment is set for MLflowLogger and callbacks

        with mlflow.start_run(experiment_id=exp.experiment_id) as run:
            # 1. Resolve dataset (triggers @cached_property)
            _ = self.dataset

            # 2 . Build datamodule (triggers @cached_property)
            _ = self.datamodule

            # 3. Build model
            _ = self.model  # triggers @cached_property to build the model

            # 4. Build callbacks & logger
            callbacks = self._build_default_callbacks() + list(self._build_callbacks())
            logger = self._build_logger(run_id=run.info.run_id)

            # 5. Build Lightning Trainer
            self._lightning_trainer = L.Trainer(
                max_epochs=self.max_epochs,
                logger=logger,
                callbacks=callbacks,
                accelerator='auto',
                **self.trainer_kwargs,
            )

            # 6. Train
            _LOGGER.info("Starting training...")
            self._lightning_trainer.fit(self.model, datamodule=self.datamodule)

            # 7. Test
            _LOGGER.info("Starting test...")
            test_results = self._lightning_trainer.test(datamodule=self.datamodule)

            # 8. Build deploy adapter (only needed when the model is not already a DatamintModel)
            adapter = None
            if self.auto_deploy_adapter and not isinstance(self.model, BaseDatamintModel):
                _LOGGER.debug("Building deploy adapter...")
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

    @cached_property
    def datamodule(self) -> DatamintDataModule:
        # Build transforms
        train_tf = self._user_train_transform or self._train_transform()
        eval_tf = self._user_eval_transform or self._eval_transform()

        datamodule = self._build_datamodule(self.dataset, train_tf, eval_tf)
        return datamodule

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

    def _build_default_callbacks(self) -> list:
        from datamint.mlflow.lightning.callbacks import MLFlowPyTorchModelCheckpoint, MLFlowDatamintModelCheckpoint
        from mlflow.pyfunc.model import PythonModel

        metric_name, mode = self._monitor_metric()
        model_name = self.register_model_name or self._project_name

        if isinstance(self.model, PythonModel):
            checkpoint_cls = MLFlowDatamintModelCheckpoint
        else:
            checkpoint_cls = MLFlowPyTorchModelCheckpoint

        _LOGGER.debug(
            f"Using {checkpoint_cls.__name__} for model checkpointing with monitor='{metric_name}' mode='{mode}'")

        callbacks: list = [
            checkpoint_cls(
                monitor=metric_name,
                mode=mode,
                save_top_k=1,
                register_model_name=model_name,
                register_model_on='test',
                log_model_metrics=True,  # TODO: move this functionality to a separate callback or here
            )]

        callbacks.append(_LogDatasetSplitsCallback(self))

        return callbacks

    def _build_callbacks(self) -> list:
        from lightning.pytorch.callbacks import EarlyStopping

        metric_name, mode = self._monitor_metric()

        callbacks = []
        if self.early_stopping_patience is not None:
            callbacks.append(EarlyStopping(
                monitor=metric_name,
                mode=mode,
                patience=self.early_stopping_patience,
            ))

        return callbacks

    def _build_logger(self, run_id: str | None = None):
        from lightning.pytorch.loggers import MLFlowLogger

        self._with_project()

        mlflow_logger = MLFlowLogger(experiment_name=self.experiment_name, run_id=run_id)
        # Injecting dataset for _BaseMLFlowModelCheckpoint:_log_test_metrics_to_model
        dataset = self.datamodule.get_mlflow_dataset_split('test')
        if dataset is None:
            dataset = self.datamodule.get_mlflow_dataset()
        mlflow_logger._mlflow_dataset = dataset
        return mlflow_logger


class _LogDatasetSplitsCallback(L.Callback):
    """Lightning callback to retrieve resolved dataset splits from the datamodule after setup()."""

    LIGHTNING_STAGE_TO_DATAMINT_SPLIT = {
        'fit': 'train',
        'validate': 'val',
        'test': 'test',
    }

    def __init__(self, dttrainer: BaseTrainer) -> None:
        super().__init__()
        self.dttrainer = dttrainer

    def setup(self, trainer: "L.Trainer", pl_module: "L.LightningModule", stage: str) -> None:

        split = self.LIGHTNING_STAGE_TO_DATAMINT_SPLIT.get(stage)
        if split is None:
            return

        mlflow_dataset = self.dttrainer.datamodule.get_mlflow_dataset_split(split)
        if mlflow_dataset is None:
            return
        try:
            _LOGGER.info(f"Logging dataset split '{split}' to MLflow for model context...")
            mlflow.log_input(mlflow_dataset, context=split)
            _LOGGER.debug(f"Successfully logged dataset split '{split}' to MLflow.")
        except Exception as e:
            _LOGGER.warning(f"Failed to log dataset input: {e}")
