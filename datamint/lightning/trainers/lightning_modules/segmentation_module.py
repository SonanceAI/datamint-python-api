"""LightningModule wrapper for segmentation tasks."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import lightning as L
import torch
from torch import Tensor, nn


class SegmentationModule(L.LightningModule):
    """Generic segmentation module backed by ``segmentation_models_pytorch``.

    Args:
        arch: SMP architecture name (e.g. ``'UnetPlusPlus'``, ``'DeepLabV3Plus'``).
        encoder_name: Backbone encoder (e.g. ``'resnet34'``).
        in_channels: Number of input channels.
        num_classes: Number of segmentation classes **excluding** background.
        loss_fn: Loss module.
        metrics_factories: ``{name: callable}`` where each callable returns a
            fresh :class:`torchmetrics.Metric`.  One instance is created per
            stage (train / val / test).
        lr: Learning rate for AdamW.
    """

    def __init__(
        self,
        arch: str,
        encoder_name: str,
        in_channels: int,
        num_classes: int,
        loss_fn: nn.Module,
        metrics_factories: dict[str, Callable[[], Any]],
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['loss_fn', 'metrics_factories'])

        import segmentation_models_pytorch as smp

        arch_cls = getattr(smp, arch)
        self.model = arch_cls(
            encoder_name=encoder_name,
            encoder_weights='imagenet',
            in_channels=in_channels,
            classes=num_classes,
        )
        self.criterion = loss_fn

        # Create per-stage metrics
        self._metric_names = list(metrics_factories.keys())
        for stage in ('train', 'val', 'test'):
            for name, factory in metrics_factories.items():
                self.add_module(f'{stage}_{name}', factory())

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _common_step(self, batch: dict, stage: str) -> Tensor:
        images = batch['image']
        masks = batch['segmentations'][:, 1:]  # exclude background channel

        logits = self(images)
        loss = self.criterion(logits, masks)
        preds = (logits > 0).long()

        for name in self._metric_names:
            getattr(self, f'{stage}_{name}').update(preds, masks.long())

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
