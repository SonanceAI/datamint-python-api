"""2-D semantic segmentation trainer."""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn

from datamint.dataset import ImageDataset

from .lightning_modules import SegmentationModule
from .segmentation_trainer import SegmentationTrainer

if TYPE_CHECKING:
    from albumentations import BaseCompose
    from datamint.entities import Project

class SemanticSegmentation2DTrainer(SegmentationTrainer):
    """Trainer for 2-D semantic segmentation.

    Default model: **UNet++** (``segmentation_models_pytorch``) with a
    ``resnet34`` encoder pretrained on ImageNet.

    Args:
        encoder_name: SMP encoder backbone.  Defaults to ``'resnet34'``.
        in_channels: Number of input image channels.  Defaults to ``3``.
        All remaining keyword arguments are forwarded to
        :class:`~datamint.lightning.trainers.base_trainer.BaseTrainer`.

    Example::

        trainer = SemanticSegmentation2DTrainer(project='BUS_Segmentation')
        results = trainer.fit()
    """

    def __init__(
        self,
        *,
        encoder_name: str = 'resnet34',
        in_channels: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.encoder_name = encoder_name
        self.in_channels = in_channels

    # ── Template hooks ──────────────────────────────────────────

    def _build_default_dataset(self, project: 'str | Project') -> ImageDataset:
        return ImageDataset(
            project=project,
            return_as_semantic_segmentation=True,
            semantic_seg_merge_strategy='union',
            allow_external_annotations=True,
            include_unannotated=False,
        )

    def _build_default_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Any],
    ) -> L.LightningModule:
        return SegmentationModule(
            arch='UnetPlusPlus',
            encoder_name=self.encoder_name,
            in_channels=self.in_channels,
            num_classes=len(self.dataset.seglabel_list),
            loss_fn=loss_fn,
            metrics_factories=metrics,
        )

    def _default_train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.Normalize(), # Imagenet stats is the default
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

    def _build_deploy_adapter(self) -> Any:
        from datamint.mlflow.flavors.model import DatamintModel
        from datamint.mlflow.flavors import datamint_flavor
        from datamint.entities.annotations import ImageSegmentation
        import mlflow

        class _SegAdapter(DatamintModel):
            """Auto-generated adapter for a trained segmentation model."""

            def __init__(self, model_name: str, class_names: list[str], image_size: tuple[int, int]):
                super().__init__(
                    mlflow_torch_models_uri={'model': f'models:/{model_name}/latest'},
                    settings={'need_gpu': True},
                )
                self._class_names = class_names
                self._image_size = image_size

            def predict_default(self, model_input, **kwargs):
                import cv2
                import numpy as np
                import torch
                import albumentations as A
                from albumentations.pytorch import ToTensorV2

                model = self.get_mlflow_torch_models()['model']
                model.eval()
                fabric = L.Fabric(accelerator=self.inference_device)
                model = fabric.setup_module(model)

                transform = A.Compose([
                    A.Resize(*self._image_size),
                    A.Normalize(),
                    ToTensorV2(),
                ])

                all_preds: list[list] = []
                with torch.inference_mode():
                    for res in model_input:
                        image = np.array(res.fetch_file_data(auto_convert=True, use_cache=True))
                        oh, ow = image.shape[:2]
                        tensor = transform(image=image)['image'].to(fabric.device)
                        logits = model(tensor.unsqueeze(0))
                        pred = (logits[0] > 0).cpu().numpy().astype(np.uint8)

                        anns: list = []
                        for i, name in enumerate(self._class_names):
                            mask = cv2.resize(
                                pred[i], (ow, oh),
                                interpolation=cv2.INTER_NEAREST,
                            ) * 255
                            if mask.any():
                                anns.append(ImageSegmentation(name=name, mask=mask))
                        all_preds.append(anns)
                return all_preds

        project_name = self.dataset.project.name if self.dataset.project else 'datamint'
        model_name = self.register_model_name or project_name
        adapter = _SegAdapter(model_name, list(self.dataset.seglabel_list), self.image_size)

        experiment_name = self.mlflow_experiment_name or f"{project_name}_training"
        mlflow.set_experiment(f'{experiment_name}_deployment')
        with mlflow.start_run(run_name='auto_adapter'):
            datamint_flavor.log_model(
                adapter,
                registered_model_name=f'{model_name}_adapted',
            )

        return adapter


class UNetPPTrainer(SemanticSegmentation2DTrainer):
    """Convenience trainer pre-configured for UNet++ with stronger augmentations.

    Adds elastic transform and grid distortion to the default training
    pipeline — augmentations that are particularly effective for medical
    image segmentation.

    Example::

        trainer = UNetPPTrainer(
            project='BUS_Segmentation',
            encoder_name='efficientnet-b4',
        )
        results = trainer.fit()
    """

    def _default_train_transform(self) -> 'BaseCompose':
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        h, w = self.image_size
        return A.Compose([
            A.Resize(h, w),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ElasticTransform(alpha=50, sigma=5, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])
