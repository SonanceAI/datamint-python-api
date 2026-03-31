"""LightningModule wrapper for segmentation tasks."""
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import Any
import inspect
import warnings

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
        if num_classes <= 0:
            raise ValueError("num_classes must be > 0")
        if class_names and (len(class_names) != num_classes):
            raise ValueError("Length of class_names must match num_classes")
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

        if self._log_sample_metrics:
            loss_unreduced = self._compute_unreduced_loss(logits, masks)
            loss = loss_unreduced.mean()
            self._accumulate_sample_data(batch, logits, loss_unreduced, stage)
        else:
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

    def _compute_unreduced_loss(self, logits: Tensor, masks: Tensor) -> Tensor:
        """Compute per-sample loss (shape ``(B,)``).

        Attempts to call ``self.criterion`` with ``reduction='none'``.
        Falls back to ``binary_cross_entropy_with_logits`` if the criterion
        does not support that parameter.
        """
        b = logits.shape[0]

        # Try criterion with reduction='none' — inspect first to avoid a
        # costly forward pass that raises an exception at runtime.
        criterion_params = inspect.signature(self.criterion.forward).parameters
        if 'reduction' in criterion_params:
            loss = self.criterion(logits, masks.float(), reduction='none')
            # Flatten spatial dims if needed so result is (B,).
            if loss.dim() > 1:
                loss = loss.reshape(b, -1).mean(dim=1)
            return loss
        else:
            warnings.warn(
                f"{type(self.criterion).__name__} does not accept reduction='none'; "
                "falling back to binary_cross_entropy_with_logits for per-sample loss.",
                UserWarning,
                stacklevel=2,
            )

        # Fallback: BCE with logits averaged over spatial dims per sample.
        logits_flat = logits.reshape(b, -1)
        masks_flat = masks.reshape(b, -1).float()
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits_flat, masks_flat, reduction='none',
        ).mean(dim=1)

    def _compute_sample_confidence(self, logits: Tensor) -> dict[str, Tensor]:
        """Sigmoid-based aggregate and per-class confidence."""
        probs = torch.sigmoid(logits)  # (B, C, H, W)
        result: dict[str, Tensor] = {
            'confidence': probs.mean(dim=[1, 2, 3]),  # (B,)
        }
        for i, name in enumerate(self.class_names):
            result[f'confidence/{name}'] = probs[:, i].mean(dim=[1, 2])
        return result

    def _compute_sample_metrics(self, logits: Tensor, batch: dict) -> dict[str, Tensor]:
        """Per-sample IoU and Dice."""
        masks = batch['segmentations'][:, 1:].float()
        preds = (logits > 0).float()
        intersection = (preds * masks).sum(dim=[1, 2, 3])
        union = ((preds + masks) > 0).float().sum(dim=[1, 2, 3])
        pred_sum = preds.sum(dim=[1, 2, 3])
        mask_sum = masks.sum(dim=[1, 2, 3])
        eps = 1e-6
        iou = (intersection + eps) / (union + eps)
        dice = (2 * intersection + eps) / (pred_sum + mask_sum + eps)
        return {'iou': iou, 'dice': dice}

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
                # Per-sample confidence
                probs = torch.sigmoid(logits)
                sample_confidence = float(probs.mean())
                pred = (logits[0] > 0).cpu().numpy().astype(np.uint8)
                anns: list = []
                for i, name in enumerate(self.class_names):
                    class_conf = float(probs[0, i].mean())
                    mask = cv2.resize(
                        pred[i], (ow, oh),
                        interpolation=cv2.INTER_NEAREST,
                    ) * 255
                    if mask.any():
                        anns.append(ImageSegmentation(
                            name=name, mask=mask,
                            confiability=class_conf,
                        ))
                all_preds.append(anns)
        return all_preds
