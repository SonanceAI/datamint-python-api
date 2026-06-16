"""Convenience trainer pre-configured for UNETR++."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TYPE_CHECKING

import lightning as L
from torch import nn
from typing_extensions import override

from ..lightning_modules import UNETRPPModule
from ..vol_seg_trainer import VolumeSegmentationTrainer

if TYPE_CHECKING:
    from datamint.dataset.base import DatamintBaseDataset
    from datamint.entities import Project
    from datamint.lightning.trainers.lightning_modules.base import DatamintLightningModule


class UNETRPPTrainer(VolumeSegmentationTrainer):
    """Convenience trainer pre-configured for UNETR++.

    UNETR++ is a true 3-D segmentation model built on a hierarchical
    transformer encoder with Efficient Paired Attention (EPA) and a CNN
    decoder with skip connections.

    Reference: Shaker et al., "UNETR++: Delving into Efficient and Accurate
    3D Medical Image Segmentation", IEEE TMI 2024.

    Example::

        trainer = UNETRPPTrainer(project='CT_Liver')
        results = trainer.fit()

    Args:
        dataset: A pre-built :class:`~datamint.dataset.base.DatamintBaseDataset`.
            Mutually exclusive with *project*.
        project: Project name or :class:`~datamint.entities.Project` object.
        patch_crop_size: ``(D, H, W)`` patch randomly cropped from each volume
            during training and used as the sliding-window patch size at eval.
            Must be divisible by 32 in each dimension.  Default ``(128, 128, 128)``.
        feature_size: Base channel width ``F``.  Encoder stage dims are
            ``[2F, 4F, 8F, 16F]``.  Default ``16`` matches the original paper.
        num_heads: Number of attention heads in EPA.  Default ``4``.
        depths: Number of transformer blocks per encoder stage (list of 4 ints).
            Default ``[3, 3, 3, 3]`` from the paper's Synapse config.
        sw_overlap: Overlap ratio for sliding-window inference.  Higher values
            reduce tiling artefacts at the cost of more compute.  Default ``0.5``.
        in_channels: Number of input image channels (e.g. ``1`` for CT,
            ``4`` for multi-modal MRI).  Default ``1``.
        batch_size: Training batch size (number of volumes).  Defaults to ``1``
            because 3-D volumes typically differ in spatial size.
        loss_fn: Custom loss.  Defaults to combined BCE + Dice.
        max_epochs: Maximum training epochs.
        early_stopping_patience: Epochs without val improvement before stop.
        mlflow_experiment_name: MLflow experiment name (auto-generated if None).
        register_model_name: MLflow model registry name (auto-generated if None).
        auto_deploy_adapter: Auto-generate a deployment adapter after training.
        trainer_kwargs: Extra kwargs forwarded to :class:`lightning.Trainer`.
        dataset_kwargs: Extra kwargs forwarded to :class:`VolumeDataset`.
    """

    def __init__(
        self,
        dataset: 'DatamintBaseDataset | None' = None,
        project: 'str | Project | None' = None,
        *,
        patch_crop_size: tuple[int, int, int] = (128, 128, 128),
        feature_size: int = 16,
        num_heads: int = 4,
        depths: list[int] | None = None,
        sw_overlap: float = 0.5,
        in_channels: int = 1,
        model: L.LightningModule | type[L.LightningModule] | None = None,
        loss_fn: nn.Module | None = None,
        batch_size: int = 1,
        num_workers: int = 4,
        split_as_of_timestamp: str | None = None,
        max_epochs: int = 1,
        early_stopping_patience: int | None = 10,
        mlflow_experiment_name: str | None = None,
        register_model_name: str | None = None,
        auto_deploy_adapter: bool = True,
        trainer_kwargs: dict[str, Any] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            dataset=dataset,
            project=project,
            patch_crop_size=patch_crop_size,
            model=model,
            loss_fn=loss_fn,
            batch_size=batch_size,
            num_workers=num_workers,
            split_as_of_timestamp=split_as_of_timestamp,
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
            mlflow_experiment_name=mlflow_experiment_name,
            register_model_name=register_model_name,
            auto_deploy_adapter=auto_deploy_adapter,
            trainer_kwargs=trainer_kwargs,
            dataset_kwargs=dataset_kwargs,
            **kwargs,
        )
        self.feature_size = feature_size
        self.num_heads = num_heads
        self.depths = depths or [3, 3, 3, 3]
        self.sw_overlap = sw_overlap
        self.in_channels = in_channels

    @override
    def _build_model(
        self,
        loss_fn: nn.Module,
        metrics: dict[str, Callable],
    ) -> 'DatamintLightningModule':
        num_classes = len(self.dataset.seglabel_list)
        if num_classes == 0:
            raise ValueError(
                "No segmentation labels found in the dataset. "
                "UNETRPPTrainer requires at least one segmentation label. "
                "Make sure the project has annotated resources with segmentation labels."
            )
        return UNETRPPModule(
            in_channels=self.in_channels,
            num_classes=num_classes,
            img_size=self.patch_crop_size,
            loss_fn=loss_fn,
            metrics_factories=metrics,
            class_names=list(self.dataset.seglabel_list),
            feature_size=self.feature_size,
            num_heads=self.num_heads,
            depths=self.depths,
            patch_crop_size=self.patch_crop_size,
            sw_overlap=self.sw_overlap,
        )
