"""LightningModule wrapper for segmentation tasks."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn

from .base import DatamintLightningModule


class SegmentationModule(DatamintLightningModule):
    """Base segmentation module for semantic segmentation tasks.

    Subclasses must implement :meth:`_build_model` to return the model.

    Args:
        in_channels: Number of input channels.
        num_classes: Number of segmentation classes **excluding** background.
        loss_fn: Loss module.
        metrics_factories: ``{name: callable}`` where each callable returns a
            fresh :class:`torchmetrics.Metric`.  One instance is created per
            stage (train / val / test).
        class_names: Human-readable label for each class.
        image_size: ``(height, width)`` used during inference.
        lr: Learning rate for AdamW.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        loss_fn: nn.Module,
        metrics_factories: dict[str, Callable[[], Any]],
        class_names: list[str],
        image_size: tuple[int, int],
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.save_hyperparameters(ignore=['loss_fn', 'metrics_factories'])
        self.class_names = class_names
        self.image_size = image_size

        self.model = self._build_model()
        self.criterion = loss_fn

        # Create per-stage metrics
        self._metric_names = list(metrics_factories.keys())
        for stage in ('train', 'val', 'test'):
            for name, factory in metrics_factories.items():
                self.add_module(f'{stage}_{name}', factory())

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Instantiate and return the model. Subclasses may access
        ``self.in_channels`` and ``self.num_classes``."""
        ...

    def forward(self, x: Tensor) -> Tensor:  # type: ignore[override]
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

    def predict_default(
        self,
        model_input,
        **kwargs: Any,
    ):
        """Run segmentation inference, returning :class:`~datamint.entities.annotations.ImageSegmentation` per resource."""
        import cv2
        import numpy as np
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        from datamint.entities.annotations import ImageSegmentation

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
                oh, ow = image.shape[:2]
                tensor = transform(image=image)['image'].to(device)
                logits = self(tensor.unsqueeze(0))
                pred = (logits[0] > 0).cpu().numpy().astype(np.uint8)
                anns: list = []
                for i, name in enumerate(self.class_names):
                    mask = cv2.resize(
                        pred[i], (ow, oh),
                        interpolation=cv2.INTER_NEAREST,
                    ) * 255
                    if mask.any():
                        anns.append(ImageSegmentation(name=name, mask=mask))
                all_preds.append(anns)
        return all_preds
