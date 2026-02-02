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

from medimgkit.readers import read_array_normalized
from .base import DatamintBaseDataset, DatamintDatasetException

_LOGGER = logging.getLogger(__name__)


class ImageDataset(DatamintBaseDataset):
    """Dataset for 2D images.

    This dataset provides simple 1:1 mapping between index and resource.
    Suitable for X-rays, pathology patches, single-frame DICOM, etc.

    Each `__getitem__` returns a single 2D image with shape (C, H, W).

    Args:
        project_name: Name of the project.
        auto_update: If True, sync with server on init.
        api_key: API key for authentication.
        server_url: Datamint server URL.
        all_annotations: If True, include unpublished annotations.
        return_metainfo: If True, include metadata in output.
        return_annotations: If True, include raw annotations in output.
        return_segmentations: If True, process and return segmentations.
        return_as_semantic_segmentation: If True, convert to semantic format.
        semantic_seg_merge_strategy: Strategy for merging multi-annotator segs.
        alb_transform: Albumentations transform (applied to image+mask together).
        include_unannotated: If True, include resources without annotations.
        include_annotators: Whitelist of annotators.
        exclude_annotators: Blacklist of annotators.
        include_segmentation_names: Whitelist of segmentation labels.
        exclude_segmentation_names: Blacklist of segmentation labels.
        include_image_label_names: Whitelist of image labels.
        exclude_image_label_names: Blacklist of image labels.

    Example:
        >>> dataset = ImageDataset("chest-xray-project")
        >>> item = dataset[0]
        >>> item['image'].shape  # (C, H, W)
        torch.Size([1, 512, 512])
        >>> item['segmentations']  # dict[author -> Tensor]
    """

    @override
    def _get_raw_item(self, index: int) -> dict[str, Any]:
        """Load raw image and metadata."""
        resource = self.resources[index]
        res_bytesdata = resource.fetch_file_data(auto_convert=False, use_cache=True)

        img, metainfo = read_array_normalized(res_bytesdata, return_metainfo=True)  # shape: (N, C, H, W)
        img = img.transpose(1, 0, 2, 3)  # (N, C, H, W) -> (C, N, H, W)
        _LOGGER.debug(f"Raw image shape from resource {resource.filename}: {img.shape}")

        if img.ndim == 4:
            if img.shape[1] > 1:
                raise DatamintDatasetException(f"2D ImageDataset with 3d image at {resource.filename} detected!")
        elif img.ndim == 3:
            # (C, H, W) - add depth dim
            img = img[:, None, ...]
        else:
            raise DatamintDatasetException(f"Unexpected image shape {img.shape} for 2D image at {resource.filename}")

        anns = self.resource_annotations[index]

        return {
            'image': img,  # shape (C, N, H, W)
            'metainfo': metainfo,
            'annotations': anns,
            'resource': resource,
        }

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        result = super().__getitem__(index)
        img = result['image']
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
        """Apply albumentations transform to image and masks.

        Args:
            img: Image array of shape (C, depth, H, W).
            segmentations: Dict of author -> list of mask arrays of shape (#instances, depth, H, W).
        Returns:
            Dict with transformed 'image' and 'segmentations'.

        """
        if self.alb_transform is None:
            raise ValueError("alb_transform is not set")
        if img.ndim != 4:
            raise ValueError(f"Expected 4D image array (C, depth, H, W), got shape {img.shape}")

        img_kw = 'image'
        mask_kw = 'masks'
        apply_per_depth_slice = True

        # transpose to (depth, H, W, C)
        img = np.transpose(img, (1, 2, 3, 0))

        replay_alb_transf = albumentations.ReplayCompose([self.alb_transform])
        _LOGGER.debug(
            f'before alb transform image shape: {img.shape} | segmentations shape: {[segmentations[a].shape for a in segmentations]}')

        # Handle 4D data by iterating over depth slices
        depth = img.shape[0]
        aug_img_slices = []
        aug_seg_slices = {author: [] for author in segmentations}
        replay_data = None

        for d in range(depth):
            img_slice = img[d]  # (H, W, C)

            # Apply transform to first slice or replay to subsequent slices
            if apply_per_depth_slice or d == 0:
                _LOGGER.debug(
                    f"Applying albumentations transform to depth slice {d}. img_slice shape: {img_slice.shape}")
                aug_data = replay_alb_transf(**{img_kw: img_slice})
                aug_img_slices.append(aug_data[img_kw])
                replay_data = aug_data['replay']
            else:
                # Replay the same transform from first slice
                aug_result = replay_alb_transf.replay(replay_data, **{img_kw: img_slice})
                aug_img_slices.append(aug_result[img_kw])

            # Apply same transform to segmentation masks for this slice
            for author, segs in segmentations.items():
                # segs shape: (#instances, depth, H, W)
                segs_slice = segs[:, d, :, :]  # (#instances, H, W)
                aug_segs_slice = replay_alb_transf.replay(replay_data, **{mask_kw: segs_slice})[mask_kw]
                aug_seg_slices[author].append(aug_segs_slice)

        # Stack slices back together
        if isinstance(aug_img_slices[0], np.ndarray):
            aug_img = np.stack(aug_img_slices, axis=0)  # (depth, H, W, C)
        else:
            aug_img = torch.stack(aug_img_slices, dim=0)  # (depth, H, W, C)
        _LOGGER.debug(f"augmented image shape after stacking: {aug_img.shape}")
        aug_segmentations = {}
        for author in segmentations:
            # Stack to get (#instances, depth, H, W)
            if isinstance(aug_seg_slices[author][0], np.ndarray):
                aug_segmentations[author] = np.stack(aug_seg_slices[author], axis=1)
            else:
                aug_segmentations[author] = torch.stack(aug_seg_slices[author], dim=1)

        # else:
            # # Apply transform to 3D image at once
            # aug_data = replay_alb_transf(**{img_kw: img})
            # aug_img = aug_data[img_kw]
            # replay_data = aug_data['replay']

            # aug_segmentations = {}
            # for author, segs in segmentations.items():
            #     segs = replay_alb_transf.replay(replay_data, **{mask_kw: segs})[mask_kw]
            #     aug_segmentations[author] = segs

        # transpose back to (C, H, W) or (C, depth, H, W)
        if isinstance(aug_img, np.ndarray):
            aug_img = np.transpose(aug_img, (3, 0, 1, 2))
        elif isinstance(aug_img, torch.Tensor):
            # shape is (depth, C, H, W), assuming albumentation transformation changed it
            _LOGGER.debug(f"augmented image tensor shape before permute: {aug_img.shape}")
            if aug_img.shape[1] == img.shape[-1]:
                aug_img = aug_img.permute(1, 0, 2, 3)
            else:
                aug_img = aug_img.permute(3, 0, 1, 2)
            _LOGGER.debug(f"augmented image tensor shape after permute: {aug_img.shape}")

        return {
            'image': aug_img,
            'segmentations': aug_segmentations,
        }

    @override
    def __repr__(self) -> str:
        base = super().__repr__()
        return f"ImageDataset\n{base}"
