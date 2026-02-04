"""
ImageDataset - Dataset for 2D images.

Handles standard 2D medical images like X-rays, pathology patches, 
single-frame DICOM, PNG, JPEG, etc.
"""
import logging
from typing import Any
from typing_extensions import override
import torch
from torch import Tensor
import numpy as np
import albumentations

from .base import DatamintDatasetException
from .volume_dataset import VolumeDataset

_LOGGER = logging.getLogger(__name__)


class ImageDataset(VolumeDataset):
    """Dataset for 2D medical images."""
    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        img = result['image']
        if img.shape[1] != 1:
            raise DatamintDatasetException("Expected 2D image with shape (C, 1, H, W)"
                                           f", got {img.shape}")
        img = img.squeeze(1)  # (C, 1, H, W) -> (C, H, W)
        result['image'] = img

        if self.return_segmentations:
            segmentations = result['segmentations']
            _LOGGER.debug(f"final segmentations type: {type(segmentations)} | shape: {segmentations.shape}")
            # convert segmentations shape to expected format:
            # if semantic and no merge: dict[author -> (num_labels+1, H, W)]
            # if instance and no merge: dict[author -> (num_instances, H, W)]
            # if merged (semantic only): (num_labels+1, H, W)
            if isinstance(segmentations, (Tensor, np.ndarray)):
                _LOGGER.debug(f"squeezing merged segmentations of shape {segmentations.shape}")
                segmentations = segmentations.squeeze(1)
            else:
                for author in segmentations:
                    segmentations[author] = segmentations[author].squeeze(1)

            result['segmentations'] = segmentations
            _LOGGER.debug(
                f"final segmentations after squeeze type: {type(segmentations)} | shape: {segmentations.shape}")

        return result

    @override
    def apply_alb_transform(
        self,
        img: np.ndarray,
        segmentations: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        if self.alb_transform is None:
            raise ValueError("alb_transform is not set")
        if img.ndim == 4:
            if img.shape[1] != 1:
                raise ValueError(f"Expected 2D image with shape (C, 1, H, W), got {img.shape}")
            img = img.squeeze(1)  # (C, 1, H, W) -> (C, H, W)
        elif img.ndim != 3:
            raise ValueError(f"Expected 3D image array (C, H, W) or (C, 1, H, W), got shape {img.shape}")

        # transpose to (H, W, C)
        img = np.transpose(img, (1, 2, 0))

        replay_alb_transf = albumentations.ReplayCompose([self.alb_transform])
        _LOGGER.debug(
            f'before alb transform image shape: {img.shape} | segmentations shape: {[segmentations[a].shape for a in segmentations]}')

        aug_data = replay_alb_transf(image=img)  # First call
        replay_data = aug_data['replay']
        aug_img = aug_data['image']

        aug_segmentations = {}
        for author, segs in segmentations.items():
            if segs.ndim == 4 and segs.shape[1] == 1:
                segs = segs.squeeze(1)  # (num_instances, 1, H, W) -> (num_instances, H, W)
            aug_segs = replay_alb_transf.replay(replay_data, masks=segs)['masks']
            # store back with original shape
            if segs.ndim == 3:
                aug_segs = aug_segs[:, np.newaxis, :, :]  # (num_instances, H, W) -> (num_instances, 1, H, W)
            aug_segmentations[author] = aug_segs

        # transpose back to (C, H, W)
        if isinstance(aug_img, np.ndarray):
            aug_img = np.transpose(aug_img, (2, 0, 1))
        elif isinstance(aug_img, torch.Tensor):
            # shape is (C, H, W), assuming albumentation transformation changed it
            _LOGGER.debug(f"augmented image tensor shape before permute: {aug_img.shape}")
            if aug_img.shape[0] == img.shape[-1]:  # if C is in dim 0
                aug_img = aug_img.permute(0, 1, 2)
            else:
                aug_img = aug_img.permute(2, 0, 1)
            _LOGGER.debug(f"augmented image tensor shape after permute: {aug_img.shape}")
        # back to (C, 1, H, W)
        aug_img = aug_img[:, np.newaxis, :, :]

        return {
            'image': aug_img,
            'segmentations': aug_segmentations,
        }

    @override
    def __repr__(self) -> str:
        base = super(VolumeDataset, self).__repr__()
        return f"ImageDataset\n{base}"
