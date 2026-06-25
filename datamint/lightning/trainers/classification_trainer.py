"""Image classification trainers."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn

from datamint.dataset import ImageDataset
from functools import partial

from datamint.entities.annotations.annotation_spec import CategoryAnnotationSpec
from datamint.entities.annotations.types import AnnotationType
from .lightning_modules import ClassificationModule
from .base_trainer import BaseTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.entities import Project


class ClassificationTrainer(BaseTrainer):
    """Abstract trainer for classification tasks.

    Provides shared defaults:

    * **Loss** – :class:`~torch.nn.CrossEntropyLoss`.
    * **Metrics** – Multiclass Accuracy and macro F1 (torchmetrics).
    * **Monitor** – ``val/accuracy`` (maximise).
    """

    def _loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _metrics(self) -> dict[str, Callable]:
        from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

        nc = len(self.dataset.image_categories_set)
        return {
            'accuracy': partial(MulticlassAccuracy, num_classes=nc),
            'f1': partial(MulticlassF1Score, num_classes=nc, average='macro'),
        }

    def _monitor_metric(self) -> tuple[str, str]:
        return 'val/accuracy', 'max'

    def _build_annotation_specs(self) -> list[CategoryAnnotationSpec]:
        groups: dict[str, list[str]] = {}
        for identifier, value in self.dataset.image_categories_set:
            groups.setdefault(identifier, []).append(value)
        return [
            CategoryAnnotationSpec(
                type=AnnotationType.CATEGORY,
                scope='image',
                identifier=ident,
                required=True,
                values=sorted(vals),
            )
            for ident, vals in groups.items()
        ]


class ImageClassificationTrainer(ClassificationTrainer):
    """Trainer for image classification tasks.

    Default model: **ResNet-34** (via ``timm``) pretrained on ImageNet.

    Args:
        model_name: ``timm`` model name.  Defaults to ``'resnet34'``.
        pretrained: Use pretrained weights.  Defaults to ``True``.
        image_size: Optional target image size ``(H, W)`` or a single int
            for square images. When omitted, the trainer keeps the original
            image size instead of forcing a resize.

    Example::

        trainer = ImageClassificationTrainer(project='ChestXray')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        model_name: str = 'resnet34',
        pretrained: bool = True,
        image_size: int | tuple[int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained
        if isinstance(image_size, int):
            self.image_size = (image_size, image_size)
        else:
            self.image_size = image_size

    # ── Template hooks ──────────────────────────────────────────

    def _build_dataset(self, project: 'str | Project', **kwargs: Any) -> ImageDataset:
        default_params = dict(
            return_segmentations=False,
            include_unannotated=False,
            image_categories_merge_strategy='mode',
            allow_external_annotations=True,
        )
        dataset_params = {**default_params, **kwargs}
        return ImageDataset(
            project=project,
            **dataset_params
        )

    def _build_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Any],
    ) -> L.LightningModule:
        return ClassificationModule(
            model_name=self.model_name,
            num_classes=len(self.dataset.image_categories_set),
            loss_fn=loss_fn,
            metrics_factories=metrics,
            class_names=list(self.dataset.image_categories_set),
            image_size=self.image_size,
            pretrained=self.pretrained,
        )

    def _build_resize_transform(self):
        import albumentations as A

        if self.image_size is None:
            return A.NoOp()
        return A.Resize(*self.image_size)

    def _train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            self._build_resize_transform(),
            A.ToRGB(),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _eval_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        return A.Compose([
            self._build_resize_transform(),
            A.ToRGB(),
            A.Normalize(),
            ToTensorV2(),
        ])
