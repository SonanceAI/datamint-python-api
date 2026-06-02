from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn
from typing_extensions import override

from ..lightning_modules import DeepLabV3PlusModule
from ..seg2d_trainer import SemanticSegmentation2DTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project
    from medimgkit import ViewPlane
    from datamint.lightning.trainers.lightning_modules.base import DatamintLightningModule


class DeepLabV3PlusTrainer(SemanticSegmentation2DTrainer):
    """Convenience trainer pre-configured for DeepLab v3+.

    Uses the ASPP-based DeepLab v3+ architecture from
    ``segmentation_models_pytorch``. The ``decoder_atrous_rates`` parameter
    controls the dilation rates of the Atrous Spatial Pyramid Pooling module,
    which is DeepLab v3+'s core multi-scale context mechanism.

    Example::

        trainer = DeepLabV3PlusTrainer(
            project='BUS_Segmentation',
            encoder_name='resnet50',
        )
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
        # DeepLab v3+ specific:
        encoder_name: str = 'resnet34',
        decoder_atrous_rates: tuple[int, int, int] = (12, 24, 36),
        **kwargs: Any,
    ) -> None:
        """
        Builds a DeepLab v3+ trainer with sensible defaults for segmentation tasks.

        Args:
            dataset: A pre-built :class:`DatamintBaseDataset`. Mutually
                exclusive with *project*.
            project: Project name or :class:`Project` object used to
                auto-build a dataset when *dataset* is ``None``.
            model: A user-provided :class:`~lightning.LightningModule`.
                When ``None`` the trainer builds a default one via
                :meth:`_build_model`.
            loss_fn: Custom loss function forwarded to the default model.
                Ignored when *model* is provided.
            batch_size: Training batch size.
            num_workers: DataLoader workers.
            train_transform: Albumentations transform for training. When
                ``None`` the trainer uses :meth:`_train_transform`.
            eval_transform: Albumentations transform for val/test. When
                ``None`` the trainer uses :meth:`_eval_transform`.
            image_size: Target image size ``(H, W)`` or a single int for
                square images. Forwarded to default transforms.
            split_as_of_timestamp: Historical timestamp used to resolve
                project-scoped dataset splits during training.
            max_epochs: Maximum number of training epochs.
            early_stopping_patience: Epochs without improvement before
                stopping. Set to ``None`` to disable early stopping.
            mlflow_experiment_name: MLflow experiment name. Auto-generated
                from the project name when ``None``.
            register_model_name: Name for MLflow Model Registry.
                Auto-generated when ``None``.
            auto_deploy_adapter: When ``True``, auto-generate a
                :class:`~datamint.mlflow.flavors.model.DatamintModel`
                adapter after training.
            trainer_kwargs: Extra keyword arguments forwarded to
                :class:`lightning.Trainer`.
            encoder_name: SMP encoder backbone name (e.g. ``'resnet34'``,
                ``'resnet50'``, ``'efficientnet-b4'``).
            decoder_atrous_rates: Dilation rates for the ASPP module.
                Controls the multi-scale receptive field sizes. Defaults to
                ``(12, 24, 36)``. Smaller values capture finer-grained context;
                larger values capture coarser context.
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
        self.decoder_atrous_rates = decoder_atrous_rates

    @override
    def _build_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Callable],
    ) -> 'DatamintLightningModule':
        num_classes = len(self.dataset.seglabel_list)
        if num_classes == 0:
            raise ValueError(
                "No segmentation labels found in the dataset. "
                "DeepLabV3Plus requires at least one segmentation label to train. "
                "Make sure your project has annotated resources with segmentation labels, "
                "or check that 'include_unannotated' is not masking all annotated data."
            )
        return DeepLabV3PlusModule(
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            num_classes=num_classes,
            loss_fn=loss_fn,
            metrics_factories=metrics,
            class_names=list(self.dataset.seglabel_list),
            image_size=self.image_size,
            decoder_atrous_rates=self.decoder_atrous_rates,
        )
