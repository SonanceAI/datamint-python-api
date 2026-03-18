"""Image classification trainers."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn

from datamint.dataset import ImageDataset

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

    def _default_loss(self) -> nn.Module:
        return nn.CrossEntropyLoss()

    def _default_metrics(self) -> dict[str, Any]:
        from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

        nc = len(self.dataset.image_categories_set)
        return {
            'accuracy': lambda: MulticlassAccuracy(num_classes=nc),
            'f1': lambda: MulticlassF1Score(num_classes=nc, average='macro'),
        }

    def _monitor_metric(self) -> tuple[str, str]:
        return 'val/accuracy', 'max'


class ImageClassificationTrainer(ClassificationTrainer):
    """Trainer for image classification tasks.

    Default model: **ResNet-34** (via ``timm``) pretrained on ImageNet.

    Args:
        model_name: ``timm`` model name.  Defaults to ``'resnet34'``.
        pretrained: Use pretrained weights.  Defaults to ``True``.

    Example::

        trainer = ImageClassificationTrainer(project='ChestXray')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        model_name: str = 'resnet34',
        pretrained: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_name = model_name
        self.pretrained = pretrained

    # ── Template hooks ──────────────────────────────────────────

    def _build_default_dataset(self, project: 'str | Project') -> ImageDataset:
        return ImageDataset(
            project=project,
            return_segmentations=False,
            include_unannotated=False,
            image_categories_merge_strategy='mode',
        )

    def _build_default_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Any],
    ) -> L.LightningModule:
        return ClassificationModule(
            model_name=self.model_name,
            num_classes=len(self.dataset.image_categories_set),
            loss_fn=loss_fn,
            metrics_factories=metrics,
            pretrained=self.pretrained,
        )

    def _default_train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(),
            ToTensorV2(),
        ])

    def _default_eval_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.Normalize(),
            ToTensorV2(),
        ])
