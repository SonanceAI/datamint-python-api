"""YOLOXTrainer — one-liner detection trainer backed by YOLOX."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn

from ..detection_trainer import DetectionTrainer
from ..lightning_modules.detection_modules import YOLOXModule

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project
    from datamint.lightning.trainers.lightning_modules.base import DatamintLightningModule


class YOLOXTrainer(DetectionTrainer):
    """One-liner trainer for anchor-free object detection using YOLOX.

    Wraps YOLOX (Apache 2.0) with sensible defaults so detection training
    requires only a project name and, optionally, a model size::

        trainer = YOLOXTrainer(project='thyroid_nodules', model_size='s')
        results = trainer.fit()

    Args:
        dataset: A pre-built :class:`~datamint.dataset.ImageDataset`.
            Mutually exclusive with *project*.
        project: Project name or :class:`~datamint.entities.Project` object.
            A :class:`~datamint.dataset.ImageDataset` is created automatically.
        model_size: YOLOX size variant — ``'nano'``, ``'tiny'``, ``'s'``,
            ``'m'``, ``'l'``, or ``'x'``.  Defaults to ``'s'``, which
            balances speed and accuracy for most medical imaging tasks.
        conf_thre: Combined objectness × class-confidence threshold applied
            during NMS at inference time.
        nms_thre: IoU threshold for non-maximum suppression.
        image_size: Target ``(H, W)`` size or a single int for square images.
            Both training and evaluation images are resized to this before
            being fed to YOLOX.
        batch_size: Training batch size.
        num_workers: DataLoader workers.
        train_transform: Custom albumentations transform for training.
            When ``None`` the trainer uses its built-in augmentation pipeline.
        eval_transform: Custom albumentations transform for val/test.
            When ``None`` the trainer uses resize + normalize.
        max_epochs: Maximum training epochs.
        early_stopping_patience: Patience for early stopping on ``val/map``.
            Set to ``None`` to disable.
        mlflow_experiment_name: MLflow experiment name.
        model_name: Name for the model in the registry.
        split_as_of_timestamp: Historical timestamp for reproducible splits.
        auto_deploy_adapter: Auto-log a deploy adapter after training.
        trainer_kwargs: Extra kwargs forwarded to :class:`lightning.Trainer`.
        dataset_kwargs: Extra kwargs forwarded to :class:`~datamint.dataset.ImageDataset`.
    """

    def __init__(
        self,
        dataset: 'DatamintBaseDataset | None' = None,
        project: 'str | Project | None' = None,
        *,
        model_size: str = 's',
        conf_thre: float = 0.25,
        nms_thre: float = 0.45,
        image_size: int | tuple[int, int] = 640,
        model: L.LightningModule | type[L.LightningModule] | None = None,
        loss_fn: nn.Module | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        train_transform: 'BaseCompose | None' = None,
        eval_transform: 'BaseCompose | None' = None,
        split_as_of_timestamp: str | None = None,
        max_epochs: int = 50,
        early_stopping_patience: int | None = 10,
        mlflow_experiment_name: str | None = None,
        model_name: str | None = None,
        auto_deploy_adapter: bool = True,
        trainer_kwargs: dict[str, Any] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset,
            project=project,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            train_transform=train_transform,
            eval_transform=eval_transform,
            split_as_of_timestamp=split_as_of_timestamp,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            mlflow_experiment_name=mlflow_experiment_name,
            model_name=model_name,
            auto_deploy_adapter=auto_deploy_adapter,
            trainer_kwargs=trainer_kwargs,
            dataset_kwargs=dataset_kwargs,
            **kwargs,
        )
        self.model_size = model_size
        self.conf_thre = conf_thre
        self.nms_thre = nms_thre
        if isinstance(image_size, int):
            self.image_size: tuple[int, int] = (image_size, image_size)
        else:
            self.image_size = image_size

    # ------------------------------------------------------------------
    # Transforms
    # ------------------------------------------------------------------

    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.0,
            clip=True,
        )
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.ToRGB(),
            A.HorizontalFlip(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    def _eval_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        bbox_params = A.BboxParams(
            format='pascal_voc',
            label_fields=['labels'],
            min_visibility=0.0,
            clip=True,
        )
        return A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.ToRGB(),
            A.Normalize(),
            ToTensorV2(),
        ], bbox_params=bbox_params)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _build_model(
        self,
        loss_fn: nn.Module | None,
        metrics: dict,
    ) -> 'DatamintLightningModule':
        num_classes = len(self.dataset.box_class_map)
        if num_classes == 0:
            raise ValueError(
                "No box annotation classes found in the dataset. "
                "YOLOXTrainer requires at least one annotated class. "
                "Make sure your project has BoxAnnotation objects with an identifier set."
            )

        class_names = sorted(self.dataset.box_class_map, key=self.dataset.box_class_map.__getitem__)
        return YOLOXModule(
            num_classes=num_classes,
            model_size=self.model_size,
            conf_thre=self.conf_thre,
            nms_thre=self.nms_thre,
            class_names=class_names,
        )
