from collections.abc import Callable
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import lightning as L
from torch import nn
from ..lightning_modules import UNetPPModule
from ..seg2d_trainer import SemanticSegmentation2DTrainer


if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project
    from medimgkit import ViewPlane
    from datamint.lightning.trainers.lightning_modules.base import DatamintLightningModule


class UNetPPTrainer(SemanticSegmentation2DTrainer):
    """Convenience trainer pre-configured for UNet++ with stronger augmentations.

    Adds elastic transform and grid distortion to the default training
    pipeline — augmentations that are particularly effective for medical
    image segmentation.

    Example::

        trainer = UNetPPTrainer(
            project='BUS_Segmentation',
            encoder_name='resnet34',)
        results = trainer.fit()
    """

    def __init__(
        self,
        dataset: 'DatamintBaseDataset | None' = None,
        project: 'str | Project | None' = None,
        *,
        image_size: int | tuple[int, int] | None = None,
        slice_axis: 'ViewPlane | int | None' = None,
        model: L.LightningModule | type[L.LightningModule] | None = None,
        in_channels: int = 3,
        loss_fn: nn.Module | None = None,
        batch_size: int = 16,
        num_workers: int = 4,
        train_transform: 'BaseCompose | None' = None,
        eval_transform: 'BaseCompose | None' = None,
        split_as_of_timestamp: str | None = None,
        max_epochs: int = 1,
        early_stopping_patience: int | None = 10,
        mlflow_experiment_name: str | None = None,
        register_model_name: str | None = None,
        auto_deploy_adapter: bool = True,
        trainer_kwargs: dict[str, Any] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        # UNet++ specific:
        encoder_name: str = 'resnet34',
        **kwargs: Any,
    ) -> None:
        """
        Builds a UNet++ trainer with sensible defaults for segmentation tasks.

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
            split_as_of_timestamp: Historical timestamp used to resolve
                project-scoped dataset splits during training. When omitted,
                the resolved project split datasets capture the current UTC
                timestamp and training lineage logs it via MLflow.
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
        super().__init__(
            dataset=dataset,
            project=project,
            model=model,
            in_channels=in_channels,
            image_size=image_size,
            slice_axis=slice_axis,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            eval_transform=eval_transform,
            split_as_of_timestamp=split_as_of_timestamp,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            mlflow_experiment_name=mlflow_experiment_name,
            register_model_name=register_model_name,
            auto_deploy_adapter=auto_deploy_adapter,
            trainer_kwargs=trainer_kwargs,
            dataset_kwargs=dataset_kwargs,
            **kwargs,
        )
        self.encoder_name = encoder_name

    @override
    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            self._build_resize_transform(),
            A.ToRGB(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _build_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Callable],
    ) -> 'DatamintLightningModule':
        return UNetPPModule(
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            num_classes=len(self.dataset.seglabel_list),
            loss_fn=loss_fn,
            metrics_factories=metrics,
            class_names=list(self.dataset.seglabel_list),
            image_size=self.image_size,
        )
