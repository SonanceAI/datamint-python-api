from collections.abc import Callable
from typing import Any, TYPE_CHECKING
from typing_extensions import override

import lightning as L
from torch import nn

from ..lightning_modules import TransUNetModule
from ..seg2d_trainer import SemanticSegmentation2DTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project
    from medimgkit import ViewPlane
    from datamint.lightning.trainers.lightning_modules.base import DatamintLightningModule


class TransUNetTrainer(SemanticSegmentation2DTrainer):
    """Convenience trainer pre-configured for TransUNet.

    Uses the R50-ViT-B/16 hybrid encoder with a Cascaded UPsampler (CUP)
    decoder from Chen et al. (2021).  The backbone is ``timm``'s
    ``vit_base_r50_s16_224``, which is a drop-in match for the architecture
    described in the paper.

    Example::

        trainer = TransUNetTrainer(
            project='BUS_Segmentation',
        )
        results = trainer.fit()
    """

    REQUIRED_IMAGE_SIZE: tuple[int, int] = (224, 224)

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
        # TransUNet-specific:
        variant: str = 'R50-ViT-B_16',
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Builds a TransUNet trainer with sensible defaults for segmentation tasks.

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
            image_size: Must be ``None`` or ``(224, 224)``.  TransUNet's
                positional embeddings are fixed for 224×224 inputs; any other
                size raises a ``ValueError``.
            slice_axis: Slice axis override for 3-D volume projects.
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
            variant: TransUNet variant.  Currently only ``'R50-ViT-B_16'``
                is supported (timm backbone ``vit_base_r50_s16_224``).
            pretrained: Load ImageNet-21k pre-trained weights from timm.
        """

        if image_size is not None:
            normalised = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
            if normalised != self.REQUIRED_IMAGE_SIZE:
                raise ValueError(
                    f"TransUNetTrainer requires image_size={self.REQUIRED_IMAGE_SIZE}, "
                    f"got {normalised}.  The backbone's positional embeddings are fixed "
                    "to 196 tokens (14×14 patches from a 224×224 input)."
                )

        super().__init__(
            dataset=dataset,
            project=project,
            model=model,
            in_channels=in_channels,
            image_size=self.REQUIRED_IMAGE_SIZE,
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
        self.variant = variant
        self.pretrained = pretrained

    @override
    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Include random rotation up to 30°.
        # Source: TransUNet paper (Chen et al., 2021)
        return A.Compose([
            self._build_resize_transform(),
            A.ToRGB(),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

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
                "TransUNetTrainer requires at least one segmentation label to train. "
                "Make sure your project has annotated resources with segmentation labels, "
                "or check that 'include_unannotated' is not masking all annotated data."
            )
        return TransUNetModule(
            in_channels=self.in_channels,
            num_classes=num_classes,
            loss_fn=loss_fn,
            metrics_factories=metrics,
            class_names=list(self.dataset.seglabel_list),
            image_size=self.image_size,
            variant=self.variant,
            pretrained=self.pretrained,
        )
