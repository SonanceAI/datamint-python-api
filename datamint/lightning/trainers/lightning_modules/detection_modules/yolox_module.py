"""YOLOXModule — Lightning module wrapping YOLOX for object detection."""
from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

from datamint.mlflow.flavors.task_type import TaskType
from ..base import DatamintLightningModule

_LOGGER = logging.getLogger(__name__)

_VALID_SIZES = ('nano', 'tiny', 's', 'm', 'l', 'x')


class YOLOXModule(DatamintLightningModule):
    """Lightning module wrapping YOLOX for anchor-free object detection.

    Args:
        num_classes: Number of object classes (excluding background).
        model_size: YOLOX size variant — one of ``'nano'``, ``'tiny'``, ``'s'``,
            ``'m'``, ``'l'``, ``'x'``.
        conf_thre: Objectness × class-confidence threshold for NMS filtering.
        nms_thre: IoU threshold for non-maximum suppression.
        lr: AdamW learning rate.
        class_names: Ordered list of class names (index → name) used to convert
            model output indices back to semantic labels in ``predict_image``.
            Must match the ``DetectionDataset._class_map`` order.
        transform: Eval-time albumentations transform applied in ``predict_image``.
            Set automatically by the trainer; excluded from saved hyperparameters
            so checkpoint serialization is not affected.
    """

    task_type = TaskType.OBJECT_DETECTION

    def __init__(
        self,
        num_classes: int,
        model_size: str = 's',
        conf_thre: float = 0.25,
        nms_thre: float = 0.45,
        lr: float = 1e-4,
        class_names: list[str] | None = None,
        transform: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(transform=transform, **kwargs)

        if model_size not in _VALID_SIZES:
            raise ValueError(f"model_size must be one of {_VALID_SIZES}, got '{model_size}'")

        self.save_hyperparameters(ignore=['class_names', 'transform'])
        self.class_names = class_names
        self.model = self._build_yolox_model(model_size, num_classes)
        self.map_metric = self._try_build_map_metric()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_yolox_model(model_size: str, num_classes: int):
        import yolox.models as _ym
        constructor = getattr(_ym, f'yolox_{model_size}')
        return constructor(num_classes=num_classes)

    @staticmethod
    def _try_build_map_metric():
        try:
            from torchmetrics.detection import MeanAveragePrecision
            return MeanAveragePrecision()
        except Exception:
            _LOGGER.warning(
                "torchmetrics MeanAveragePrecision unavailable (pycocotools missing?). "
                "val/map will not be logged."
            )
            return None

    # ------------------------------------------------------------------
    # Target format conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _build_targets(
        boxes_list: list[Tensor],
        labels_list: list[Tensor],
        device: torch.device,
    ) -> Tensor:
        """Convert list of Pascal-VOC boxes to YOLOX target tensor.

        Args:
            boxes_list: List of ``(N_i, 4)`` tensors in ``[x1, y1, x2, y2]`` pixels.
            labels_list: List of ``(N_i,)`` int64 class-index tensors.

        Returns:
            ``(B, max_labels, 5)`` float32 tensor where each row is
            ``[cls, cx, cy, w, h]`` in pixel coordinates.
        """
        B = len(boxes_list)
        # max_labels >= 1 so YOLOX never sees a zero-length target dimension
        max_labels = max((b.shape[0] for b in boxes_list), default=0)
        max_labels = max(max_labels, 1)
        targets = torch.zeros(B, max_labels, 5, dtype=torch.float32, device=device)
        for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
            n = boxes.shape[0]
            if n == 0:
                continue
            boxes = boxes.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            targets[i, :n, 0] = labels
            targets[i, :n, 1] = cx
            targets[i, :n, 2] = cy
            targets[i, :n, 3] = w
            targets[i, :n, 4] = h
        return targets

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch: dict, batch_idx: int) -> Tensor:
        images: Tensor = batch['image']
        targets = self._build_targets(batch['boxes'], batch['labels'], images.device)
        outputs: dict = self.model(images, targets)
        self.log('train/loss', outputs['total_loss'], prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/iou_loss', outputs['iou_loss'], on_step=False, on_epoch=True)
        self.log('train/cls_loss', outputs['cls_loss'], on_step=False, on_epoch=True)
        self.log('train/obj_loss', outputs['conf_loss'], on_step=False, on_epoch=True)
        return outputs['total_loss']

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        from yolox.utils import postprocess as yolox_postprocess

        images: Tensor = batch['image']
        with torch.no_grad():
            raw = self.model(images)
        preds = yolox_postprocess(raw, self.hparams['num_classes'],
                                  self.hparams['conf_thre'], self.hparams['nms_thre'])

        if self.map_metric is None:
            return

        pred_list = []
        target_list = []
        for i, pred in enumerate(preds):
            if pred is None:
                pred_list.append({
                    'boxes': torch.zeros((0, 4), device='cpu'),
                    'scores': torch.zeros(0, device='cpu'),
                    'labels': torch.zeros(0, dtype=torch.int64, device='cpu'),
                })
            else:
                pred = pred.cpu()
                pred_list.append({
                    'boxes': pred[:, :4],
                    'scores': pred[:, 4] * pred[:, 5],
                    'labels': pred[:, 6].long(),
                })
            target_list.append({
                'boxes': batch['boxes'][i].cpu(),
                'labels': batch['labels'][i].cpu(),
            })

        self.map_metric.update(pred_list, target_list)

    def on_validation_epoch_end(self) -> None:
        if self.map_metric is None:
            return
        result = self.map_metric.compute()
        self.log('val/map', result['map'], prog_bar=True)
        self.map_metric.reset()

    def test_step(self, batch: dict, batch_idx: int) -> None:
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self) -> None:
        if self.map_metric is None:
            return
        result = self.map_metric.compute()
        self.log('test/map', result['map'], prog_bar=True)
        self.map_metric.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=1e-4,
        )

    # ------------------------------------------------------------------
    # Inference (MLflow deploy adapter)
    # ------------------------------------------------------------------

    def predict_image(self, model_input: Any, **kwargs: Any) -> list:
        """Run detection on a list of resources and return boxes per resource.

        Args:
            model_input: List of :class:`~datamint.entities.resources.BaseResource`.

        Returns:
            ``list[list[BoxAnnotation]]`` — one inner list per resource.
        """
        import numpy as np
        from albumentations.pytorch import ToTensorV2
        from yolox.utils import postprocess as yolox_postprocess
        from datamint.entities.annotations import BoxAnnotation

        self.eval()
        all_preds: list[list[BoxAnnotation]] = []

        with torch.no_grad():
            for res in model_input:
                img_np = np.array(res.fetch_file_data(auto_convert=True, use_cache=True))

                if self.transform is not None:
                    aug = self.transform(image=img_np, bboxes=[], labels=[], identifiers=[])
                    tensor: Tensor = aug['image'].to(self.device)
                else:
                    tensor = ToTensorV2()(image=img_np)['image'].float().to(self.device)

                raw = self.model(tensor.unsqueeze(0))
                preds = yolox_postprocess(raw, self.hparams['num_classes'],
                                          self.hparams['conf_thre'], self.hparams['nms_thre'])

                boxes: list[BoxAnnotation] = []
                if preds[0] is not None:
                    for det in preds[0]:
                        x1, y1, x2, y2 = det[:4].tolist()
                        cls_idx = int(det[6].item())
                        # Resolve integer index back to the semantic class name.
                        if self.class_names and cls_idx < len(self.class_names):
                            class_name = self.class_names[cls_idx]
                        else:
                            class_name = str(cls_idx)
                        boxes.append(BoxAnnotation.from_points(
                            point1=(x1, y1),
                            point2=(x2, y2),
                            identifier=class_name,
                        ))
                all_preds.append(boxes)

        return all_preds
