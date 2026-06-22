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
            masks = result['masks']
            # convert masks shape to expected format:
            # if semantic and no merge: dict[author -> (num_labels+1, H, W)]
            # if instance and no merge: dict[author -> (num_instances, H, W)]
            # if merged (semantic only): (num_labels+1, H, W)
            if isinstance(masks, (Tensor, np.ndarray)):
                _LOGGER.debug("squeezing merged masks of shape %s", masks.shape)
                masks = masks.squeeze(1)
            else:
                for author in masks:
                    masks[author] = masks[author].squeeze(1)

            result['masks'] = masks

        return result

    @override
    def apply_alb_transform(
        self,
        img: np.ndarray,
        targets: dict[str, Any],
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
        img_hwc = np.transpose(img, (1, 2, 0))

        # Flatten masks from all authors into a single list for one transform call
        segmentations = targets.get('masks', {})
        author_order = list(segmentations.keys())
        author_counts: dict[str, int] = {}
        all_masks: list[np.ndarray] = []
        for author in author_order:
            segs = segmentations[author]
            if segs.ndim == 4 and segs.shape[1] == 1:
                segs = segs.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
            author_counts[author] = len(segs)
            all_masks.extend(list(segs))

        alb_kwargs: dict[str, Any] = {'image': img_hwc}
        if all_masks:
            alb_kwargs['masks'] = all_masks

        boxes_tensor = targets.get('boxes')
        box_labels_tensor = targets.get('box_labels')
        if boxes_tensor is not None:
            alb_kwargs['bboxes'] = boxes_tensor.tolist() if isinstance(boxes_tensor, torch.Tensor) else list(boxes_tensor)
            # Use 'labels': the standard albumentations label_fields key convention.
            alb_kwargs['labels'] = box_labels_tensor.tolist() if isinstance(box_labels_tensor, torch.Tensor) else list(box_labels_tensor)

        aug = self.alb_transform(**alb_kwargs)
        aug_img = aug['image']
        result: dict[str, Any] = {}

        # Reconstruct per-author masks
        if all_masks:
            aug_masks_flat = aug['masks']
            start = 0
            aug_segmentations: dict[str, np.ndarray] = {}
            for author in author_order:
                count = author_counts[author]
                segs_aug = np.array(aug_masks_flat[start:start + count])
                segs_aug = segs_aug[:, np.newaxis, :, :]  # (N, H, W) -> (N, 1, H, W)
                aug_segmentations[author] = segs_aug
                start += count
            result['masks'] = aug_segmentations

        # Reconstruct boxes, map 'labels' back to 'box_labels' in our output dict
        if boxes_tensor is not None:
            aug_bboxes: list = list(aug.get('bboxes', []))
            aug_box_labels: list = list(aug.get('labels', []))
            if aug_bboxes:
                result['boxes'] = torch.tensor(aug_bboxes, dtype=torch.float32)
                result['box_labels'] = torch.tensor(aug_box_labels, dtype=torch.int64)
            else:
                result['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
                result['box_labels'] = torch.zeros((0,), dtype=torch.int64)

        # transpose back to (C, H, W)
        if isinstance(aug_img, np.ndarray):
            aug_img = np.transpose(aug_img, (2, 0, 1))
        elif isinstance(aug_img, torch.Tensor):
            if aug_img.shape[0] == img_hwc.shape[-1]:  # C already in dim 0
                aug_img = aug_img.permute(0, 1, 2)
            else:
                aug_img = aug_img.permute(2, 0, 1)
        # back to (C, 1, H, W)
        aug_img = aug_img[:, np.newaxis, :, :]
        result['image'] = aug_img
        return result

    @override
    def __repr__(self) -> str:
        base = super(VolumeDataset, self).__repr__()
        return f"ImageDataset\n{base}"


def detection_collate_fn(batch: list[dict]) -> dict:
    """Collate a list of detection items into a batch.

    Images are stacked into a single tensor. Boxes and box_labels are kept as
    lists because each image may have a different number of annotations.
    """
    import torch
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'box_labels': [item['box_labels'] for item in batch],
    }
