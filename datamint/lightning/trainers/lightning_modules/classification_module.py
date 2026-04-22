"""LightningModule wrapper for image classification tasks."""
from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Any
import warnings

import albumentations as A
import torch
from torch import Tensor, nn
from torchmetrics import MetricCollection
import numpy as np
from albumentations.pytorch import ToTensorV2
from datamint.entities.annotations import ImageClassification
from datamint.mlflow.flavors.task_type import TaskType
from .base import DatamintLightningModule


class ClassificationModule(DatamintLightningModule):
    task_type = TaskType.IMAGE_CLASSIFICATION
    """Generic image classification module backed by ``timm``.

    Args:
        model_name: ``timm`` model name (e.g. ``'resnet34'``, ``'efficientnet_b0'``).
        num_classes: Number of output classes.
        loss_fn: Loss module.
        metrics_factories: ``{name: callable}`` – see :class:`SegmentationModule`.
        lr: Learning rate for AdamW.
        pretrained: Use pretrained weights.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        loss_fn: nn.Module,
        metrics_factories: dict[str, Callable[[], Any]],
        class_names: list[str],
        image_size: tuple[int, int],
        lr: float = 1e-4,
        pretrained: bool = True,
        transform: A.BasicTransform | A.BaseCompose | None = None,
    ) -> None:
        super().__init__(transform=transform)
        self.save_hyperparameters(ignore=['loss_fn', 'metrics_factories'])
        self.class_names = class_names
        self.image_size = image_size

        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.criterion = loss_fn

        _metrics = MetricCollection({name: factory() for name, factory in metrics_factories.items()})
        self.train_metrics = _metrics.clone(prefix='train/')
        self.val_metrics = _metrics.clone(prefix='val/')
        self.test_metrics = _metrics.clone(prefix='test/')

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _compute_unreduced_loss(self, logits: Tensor, labels: Tensor) -> Tensor:
        """Compute per-sample loss without reduction."""
        if not hasattr(self, '_criterion_supports_reduction_none'):
            self._criterion_supports_reduction_none = 'reduction' in inspect.signature(self.criterion.forward).parameters
            if not self._criterion_supports_reduction_none:
                warnings.warn(
                    f"Loss function {self.criterion.__class__.__name__} does not support 'reduction' argument; "
                    "per-sample logging will be inaccurate.",
                )

        if self._criterion_supports_reduction_none:
            return self.criterion(logits, labels, reduction='none')
        else:
            return self.criterion(logits, labels).unsqueeze(0).expand(logits.shape[0])

    def _common_step(self, batch: dict, stage: str) -> Tensor:
        images = batch['image']
        labels = batch['image_categories']
        logits = self(images)

        if self._log_sample_metrics:
            loss_unreduced = self._compute_unreduced_loss(logits, labels) # (B,)
            loss = loss_unreduced.mean()
            self._accumulate_sample_data(batch, logits, loss_unreduced, stage)
        else:
            loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        getattr(self, f'{stage}_metrics').update(preds, labels)

        self.log(
            f'{stage}/loss', loss,
            on_step=(stage == 'train'), on_epoch=True,
            prog_bar=True, batch_size=len(images),
        )
        return loss

    def _compute_sample_confidence(self, logits: Tensor) -> dict[str, Tensor]:
        """Softmax-based confidence: max probability and per-class probabilities."""
        probs = torch.softmax(logits, dim=1)  # (B, C)
        result: dict[str, Tensor] = {
            'confidence': probs.max(dim=1).values,  # (B,)
        }
        for i, name in enumerate(self.class_names):
            result[f'confidence/{name}'] = probs[:, i]
        return result

    def _compute_sample_metrics(self, logits: Tensor, batch: dict) -> dict[str, Tensor]:
        """Per-sample accuracy (1.0 if correct, 0.0 otherwise)."""
        labels = batch['image_categories']
        correct = (logits.argmax(dim=1) == labels).float()
        return {'accuracy': correct}

    def _on_epoch_end(self, stage: str) -> None:
        metric_col = getattr(self, f'{stage}_metrics')
        self.log_dict(metric_col.compute(), prog_bar=True)
        metric_col.reset()

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._common_step(batch, 'train')

    def validation_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._common_step(batch, 'val')

    def test_step(self, batch: dict, batch_idx: int) -> Tensor:
        return self._common_step(batch, 'test')

    def on_train_epoch_end(self) -> None:
        self._on_epoch_end('train')

    def on_validation_epoch_end(self) -> None:
        self._on_epoch_end('val')

    def on_test_epoch_end(self) -> None:
        self._on_epoch_end('test')
        super().on_test_epoch_end()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=1e-4,
        )

    def predict_default(
        self,
        model_input,
        **kwargs: Any,
    ):
        """Run classification inference, returning :class:`~datamint.entities.annotations.ImageClassification` per resource."""

        transform = self.transform
        if transform is None:
            transform = A.Compose([
                A.Resize(*self.image_size),
                A.Normalize(),
                ToTensorV2(),
            ])
        device = self.inference_device
        self.eval()
        all_preds: list[list] = []
        with torch.inference_mode():
            for res in model_input:
                image = np.array(res.fetch_file_data(auto_convert=True, use_cache=True))
                tensor = transform(image=image)['image'].to(device)
                logits = self(tensor.unsqueeze(0))
                # Per-sample confidence
                probs = torch.softmax(logits, dim=1)
                confidence = float(probs.max(dim=1).values.item())
                pred_idx = int(logits.argmax(dim=1).item())
                class_name = self.class_names[pred_idx]
                all_preds.append([ImageClassification(
                    name='category', value=class_name,
                    confiability=confidence,
                )])
        return all_preds
