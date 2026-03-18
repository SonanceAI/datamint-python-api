"""LightningModule wrapper for image classification tasks."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn


class ClassificationModule(L.LightningModule):
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
        lr: float = 1e-4,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'metrics_factories'])

        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
        self.criterion = loss_fn

        self._metric_names = list(metrics_factories.keys())
        for stage in ('train', 'val', 'test'):
            for name, factory in metrics_factories.items():
                self.add_module(f'{stage}_{name}', factory())

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _common_step(self, batch: dict, stage: str) -> Tensor:
        images = batch['image']
        labels = batch['image_categories']
        logits = self(images)
        loss = self.criterion(logits, labels)

        preds = logits.argmax(dim=1)
        for name in self._metric_names:
            getattr(self, f'{stage}_{name}').update(preds, labels)

        self.log(
            f'{stage}/loss', loss,
            on_step=(stage == 'train'), on_epoch=True,
            prog_bar=True, batch_size=len(images),
        )
        return loss

    def _on_epoch_end(self, stage: str) -> None:
        for i, name in enumerate(self._metric_names):
            metric = getattr(self, f'{stage}_{name}')
            self.log(f'{stage}/{name}', metric.compute(), prog_bar=(i == 0))
            metric.reset()

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

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=1e-4,
        )
