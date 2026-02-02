"""
VolumeDataset - Dataset for 3D medical volumes.

Handles NIfTI volumes, DICOM series, and other 3D medical imaging data
with support for different slice orientations and affine preservation.
"""
import logging
from typing import Any
from typing_extensions import override

import torch
import numpy as np
import albumentations

from medimgkit.readers import read_array_normalized
from .base import DatamintBaseDataset

_LOGGER = logging.getLogger(__name__)


# Axis mapping for anatomical orientations
SLICE_AXIS_MAP = {
    'axial': 0,      # slicing along depth (superior-inferior)
    'coronal': 1,    # slicing along height (anterior-posterior)
    'sagittal': 2,   # slicing along width (left-right)
}


class VolumeDataset(DatamintBaseDataset):
    """Dataset for 3D medical volumes.

    Handles NIfTI (3D/4D), DICOM series, and other volumetric data.
    """

    @override
    def _get_raw_item(self, index: int) -> dict[str, Any]:
        """Load raw image and metadata."""
        resource = self.resources[index]
        res_bytesdata = resource.fetch_file_data(auto_convert=False, use_cache=True)

        img, metainfo = read_array_normalized(res_bytesdata, return_metainfo=True)  # shape: (N, C, H, W)
        img = img.transpose(1, 0, 2, 3)  # (N, C, H, W) -> (C, N, H, W)
        _LOGGER.debug(f"Raw image shape from resource {resource.filename}: {img.shape}")

        anns = self.resource_annotations[index]

        return {
            'image': img,  # shape (C, N, H, W)
            'metainfo': metainfo,
            'annotations': anns,
            'resource': resource,
        }

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

        # transpose to (depth, H, W, C)
        img = np.transpose(img, (1, 2, 3, 0))

        replay_alb_transf = albumentations.ReplayCompose([self.alb_transform])
        _LOGGER.debug(
            f'before alb transform image shape: {img.shape} | segmentations shape: {[segmentations[a].shape for a in segmentations]}')

        aug_data = replay_alb_transf(volume=img)  # First call
        replay_data = aug_data['replay']
        aug_img = aug_data['volume']

        aug_segmentations = {}
        for author, segs in segmentations.items():
            aug_segmentations_author = segs.copy() if isinstance(segs, np.ndarray) else segs.clone()
            for i, seg_inst in enumerate(segs):  # for each instance mask
                aug_segmentations_author[i] = replay_alb_transf.replay(replay_data, mask3d=seg_inst)['mask3d']
            aug_segmentations[author] = aug_segmentations_author

        # transpose back to (C, H, W) or (C, depth, H, W)
        if isinstance(aug_img, np.ndarray):
            aug_img = np.transpose(aug_img, (3, 0, 1, 2))
        elif isinstance(aug_img, torch.Tensor):
            # shape is (depth, C, H, W), assuming albumentation transformation changed it
            _LOGGER.debug(f"augmented image tensor shape before permute: {aug_img.shape}")
            if aug_img.shape[1] == img.shape[-1]:  # if C is in dim 1
                aug_img = aug_img.permute(1, 0, 2, 3)
            else:
                aug_img = aug_img.permute(3, 0, 1, 2)
            _LOGGER.debug(f"augmented image tensor shape after permute: {aug_img.shape}")

        return {
            'image': aug_img,
            'segmentations': aug_segmentations,
        }

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"VolumeDataset\n{base}"
