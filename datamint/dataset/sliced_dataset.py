"""
SlicedVolumeDataset - 2D dataset created by slicing a VolumeDataset along an axis.

Provides a way to iterate over individual 2D slices from 3D volume data,
enabling training of 2D models on volumetric medical imaging data.
"""
import gzip
import logging
from functools import cached_property
from typing import Any, Literal, TYPE_CHECKING
from typing_extensions import override
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
import albumentations

from medimgkit.readers import read_array_normalized
from medimgkit import dicom_utils
from medimgkit import nifti_utils

from .base import DatamintBaseDataset
from .annotation_processor import AnnotationProcessor, MergeStrategy

from datamint.entities.cache_manager import CacheManager
if TYPE_CHECKING:
    from datamint.entities import Annotation, Resource

_LOGGER = logging.getLogger(__name__)

# Cache key for parsed slice numpy arrays
_SLICE_ARRAY_CACHEKEY = "slice_array"


class SlicedVolumeResource:
    """Proxy that presents a single 2D slice of a 3D volume Resource.

    This class wraps a :class:`Resource` and represents a specific 2D slice
    along a given axis. It uses gzip-compressed ``.npy.gz`` files on disk
    for efficient storage, with an in-memory LRU cache managed by
    :class:`CacheManager`.

    The CacheManager memory cache is disabled by default globally, but the
    sliced-volume cache manager enables it by default.

    This shared cache avoids repeated gzip decompression for already-cached
    slices. Full-volume caching is intentionally not handled here.

    Args:
        parent: The original 3D volume Resource.
        slice_index: The index of the slice along the given axis.
        slice_axis: The spatial axis to slice along ('axial', 'coronal', 'sagittal').
        sliced_vols_cache: Shared :class:`CacheManager` for disk-based volume caching.
    """

    _CACHE_MANAGER_NAMESPACE = "sliced_volumes"

    def __init__(
        self,
        parent: 'Resource',
        slice_index: int,
        slice_axis: str,
        sliced_vols_cache: CacheManager | None = None,
    ):
        self._parent = parent
        self.slice_index = slice_index
        self.slice_axis = slice_axis
        if sliced_vols_cache is None:
            sliced_vols_cache = CacheManager(SlicedVolumeResource._CACHE_MANAGER_NAMESPACE)
        self._volume_cache = sliced_vols_cache

    @staticmethod
    def slice_over(resource: 'Resource',
                   slice_axis: Literal['axial', 'coronal', 'sagittal'],
                   volume_cache: CacheManager | None = None) -> list['SlicedVolumeResource']:
        sliced_resources = []
        res_data = resource.fetch_file_data(auto_convert=True, use_cache=True)
        if resource.is_dicom():
            axis_size = dicom_utils.get_dim_size(res_data, slice_axis)
        elif resource.is_nifti():
            axis_size = nifti_utils.get_dim_size(res_data, slice_axis)
        else:
            raise ValueError(f"Unsupported resource type for slicing axis: {resource.filename}|{resource.mimetype}")

        # anns = resource_annotations[i]

        for s in range(axis_size):
            sliced_resources.append(
                SlicedVolumeResource(resource, s, slice_axis, volume_cache)
            )

        return sliced_resources

    def __getattr__(self, name: str) -> Any:
        """Delegate all unresolved attributes to the parent Resource."""
        return getattr(self._parent, name)

    def get_depth(self) -> int:
        """A single slice has depth 1."""
        return 1

    def _get_version_info(self) -> dict:
        """Get version info from the parent resource for cache validation."""
        return {
            'created_at': self._parent.created_at,
            'deleted_at': self._parent.deleted_at,
            'size': self._parent.size,
        }

    def _slice_cache_entity_id(self) -> str:
        return f"{self._parent.id}:axis{self.slice_axis}:slice{self.slice_index}"

    def fetch_slice_data(self) -> np.ndarray:
        """Fetch the 2D slice as a (C, H, W) array.

        Returns:
            Slice array with shape (C, H, W).
        """
        version_info = self._get_version_info()
        cache_entity_id = self._slice_cache_entity_id()

        cached_slice = self._volume_cache.get(
            cache_entity_id,
            _SLICE_ARRAY_CACHEKEY,
            version_info,
        )
        if cached_slice is not None:
            return np.ascontiguousarray(cached_slice)

        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        vol, self.data_metainfo = read_array_normalized(raw, return_metainfo=True)
        # vol.shape is (D,C,H,W)

        _LOGGER.debug(
            f"Slicing {self._parent.filename} along axis {self.slice_axis=} ({self.slice_axis_idx_std=}) at index {self.slice_index=} with shape {vol.shape=}")
        sliced = np.take(vol, self.slice_index, axis=self.slice_axis_idx_std)
        # vol is (D, C, H, W); after np.take the sliced axis is removed.
        # C was at index 1. If we sliced axis 0 (D), C shifts to index 0 — already correct.
        # If we sliced axis 2 (H) or 3 (W), C stays at index 1 → move it to 0.
        channel_axis_in_result = 1 if self.slice_axis_idx_std > 1 else 0
        if channel_axis_in_result != 0:
            sliced = np.moveaxis(sliced, channel_axis_in_result, 0)
        # sliced is now (C, DIM1, DIM2)
        sliced = np.ascontiguousarray(sliced)

        gz_path = self._volume_cache.get_expected_path(cache_entity_id, _SLICE_ARRAY_CACHEKEY)
        gz_path = gz_path.with_suffix('.npy.gz')
        gz_path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(str(gz_path), 'wb', compresslevel=4) as f:
            np.save(f, sliced)

        self._volume_cache.register_file_location(
            cache_entity_id,
            _SLICE_ARRAY_CACHEKEY,
            file_path=gz_path,
            version_info=version_info,
            mimetype='application/gzip',
            data=sliced,
        )

        _LOGGER.debug(f'Sliced shape: {sliced.shape}')

        return sliced

    @cached_property
    def data_metainfo(self) -> dict:
        """Volume metadata. Loaded once and cached for the lifetime of this resource."""
        raw = self._parent.fetch_file_data(auto_convert=False, use_cache=True)
        _, metainfo = read_array_normalized(raw, return_metainfo=True)
        return metainfo

    @cached_property
    def slice_axis_idx(self) -> int:
        """Raw axis index for the slice axis. Computed once and cached."""
        if self._parent.is_dicom():
            return dicom_utils.get_plane_axis(self.data_metainfo, self.slice_axis)
        elif self._parent.is_nifti():
            return nifti_utils.get_plane_axis(self.data_metainfo, self.slice_axis)
        else:
            raise ValueError(
                "Unsupported resource type for slicing axis:"
                f" {self._parent.filename}|{self._parent.mimetype}|{self.slice_axis}")

    @cached_property
    def slice_axis_idx_std(self) -> int:
        """Standardised axis index for the slice axis. Computed once and cached."""
        if self._parent.is_dicom():
            return dicom_utils.rawplaneaxis2stdplaneaxis_idx(self.slice_axis_idx)
        elif self._parent.is_nifti():
            return nifti_utils.rawplaneaxis2stdplaneaxis_idx(self.slice_axis_idx)
        else:
            raise ValueError(
                f"Unsupported resource type for slicing axis: {self._parent.filename}|{self._parent.mimetype}")

    @property
    def parent_resource(self) -> 'Resource':
        """The original volume Resource being proxied."""
        return self._parent

    def __repr__(self) -> str:
        axis_names = {0: 'axial', 1: 'coronal', 2: 'sagittal'}
        axis_name = axis_names.get(self.slice_axis, str(self.slice_axis))
        return (
            f"SlicedVolumeResource(filename='{self._parent.filename}', "
            f"axis='{axis_name}', slice={self.slice_index})"
        )

    def __getattribute__(self, name: str) -> Any:
        try:
            return super().__getattribute__(name)
        except AttributeError:
            parent = super().__getattribute__('_parent')
            return getattr(parent, name)


# # Axis mapping for anatomical orientations
# SLICE_AXIS_MAP = {
#     'axial': 0,      # slicing along depth (superior-inferior)
#     'coronal': 1,    # slicing along height (anterior-posterior)
#     'sagittal': 2,   # slicing along width (left-right)
# }

# _AXIS_INT_TO_NAME = {v: k for k, v in SLICE_AXIS_MAP.items()}


class SlicedVolumeDataset(DatamintBaseDataset):
    """2D dataset created by slicing a VolumeDataset along an axis.

    Each item corresponds to a single 2D slice from a 3D volume.
    The ``__getitem__`` returns arrays with shape ``(C, H, W)`` for images
    and ``(num_instances, H, W)`` or ``(num_labels+1, H, W)`` for segmentations.

    Typically created via :meth:`VolumeDataset.slice`, but can also be
    instantiated directly.

    Args:
        parent_dataset: The source :class:`DatamintBaseDataset` (e.g. VolumeDataset)
            providing resources, annotations, and configuration.
        slice_axis: Slice orientation. One of ``'axial'`` (depth), ``'coronal'``
            (height), ``'sagittal'`` (width), or an integer axis index (0--2).
    """

    def __init__(
        self,
        parent_dataset: 'DatamintBaseDataset',
        slice_axis: Literal['axial', 'coronal', 'sagittal'] | int = 'axial',
    ):
        # We intentionally do NOT call super().__init__() because that
        # requires project/API interaction. Instead, copy needed state
        # from the parent dataset.

        # --- Resolve axis ---
        if isinstance(slice_axis, str):
            valid_slice_axis = ['axial', 'coronal', 'sagittal']
            if slice_axis not in valid_slice_axis:
                raise ValueError(
                    f"Unknown axis '{slice_axis}'. "
                    f"Must be one of {valid_slice_axis} or an int 0-2."
                )
            self._slice_axis = slice_axis
        else:
            if not (0 <= slice_axis <= 2):
                raise ValueError(f"axis must be 0, 1, or 2, got {slice_axis}")
            self._slice_axis_int = slice_axis

        self.project = parent_dataset.project

        # Copy configuration from parent
        self.return_metainfo = parent_dataset.return_metainfo
        self.return_segmentations = parent_dataset.return_segmentations
        self.return_as_semantic_segmentation = parent_dataset.return_as_semantic_segmentation
        self.semantic_seg_merge_strategy: MergeStrategy | None = parent_dataset.semantic_seg_merge_strategy
        self.include_unannotated = parent_dataset.include_unannotated

        # Transforms
        self.alb_transform = parent_dataset.alb_transform

        # Filtering (already applied on parent's annotations)
        self.include_annotators = parent_dataset.include_annotators
        self.exclude_annotators = parent_dataset.exclude_annotators
        self.include_segmentation_names = parent_dataset.include_segmentation_names
        self.exclude_segmentation_names = parent_dataset.exclude_segmentation_names
        self.include_image_label_names = parent_dataset.include_image_label_names
        self.exclude_image_label_names = parent_dataset.exclude_image_label_names
        self.include_frame_label_names = parent_dataset.include_frame_label_names
        self.exclude_frame_label_names = parent_dataset.exclude_frame_label_names

        # Copy label sets and processor from parent
        self.annotation_processor = parent_dataset.annotation_processor
        self.frame_lsets = parent_dataset.frame_lsets
        self.frame_lcodes = parent_dataset.frame_lcodes
        self.image_lsets = parent_dataset.image_lsets
        self.image_lcodes = parent_dataset.image_lcodes
        self.seglabel_list = parent_dataset.seglabel_list
        self.seglabel2code = parent_dataset.seglabel2code

        # Internal state
        self._logged_uint16_conversion = False

        # --- Build sliced resources ---
        volume_cache = CacheManager(
            'sliced_volumes',
            enable_memory_cache=True,
            memory_cache_maxsize=2,
        )
        expanded_resources, expanded_annotations = self._expand_resources(
            parent_dataset.resources,
            parent_dataset.resource_annotations,
            volume_cache,
        )
        self.resources = expanded_resources  # type: ignore[assignment]
        self.resource_annotations = expanded_annotations

    def _expand_resources(
        self,
        resources: Sequence['Resource'],
        resource_annotations: Sequence[Sequence['Annotation']],
        volume_cache: CacheManager,
    ) -> tuple[list[SlicedVolumeResource], list[Sequence['Annotation']]]:
        """Expand volume resources into per-slice proxy resources.

        Args:
            resources: Original volume resources.
            resource_annotations: Parallel annotation sequences.
            volume_cache: Shared LRU cache for parsed volumes.

        Returns:
            Tuple of (sliced_resources, sliced_annotations).
        """
        sliced_resources: list[SlicedVolumeResource] = []
        sliced_annotations: list[Sequence['Annotation']] = []

        for i, r in enumerate(resources):
            anns = resource_annotations[i]
            per_slice = SlicedVolumeResource.slice_over(r, self._slice_axis, volume_cache)
            sliced_resources.extend(per_slice)
            sliced_annotations.extend(anns for _ in per_slice)

        return sliced_resources, sliced_annotations

    @override
    def _get_raw_item(self, index: int) -> dict[str, Any]:
        """Load a single 2D slice and its annotations.

        Returns dict with:
        - 'image': np.ndarray of shape (C, 1, DIM1, DIM2) — depth=1 to match pipeline expectations.
        - 'metainfo': dict with volume metadata.
        - 'annotations': Sequence of Annotation objects.
        - 'resource': The SlicedVolumeResource proxy.
        """
        resource: SlicedVolumeResource = self.resources[index]  # type: ignore[assignment]
        img = resource.fetch_slice_data()  # ndims=3.

        # # Add depth dim to match the pipeline expectation: (C, 1, DIM1, DIM2)
        # img = np.expand_dims(img, axis=1)

        anns = self.resource_annotations[index]

        return {
            'image': img,  # shape: (C, 1, DIM1, DIM2)
            'annotations': anns,
            'resource': resource,
        }

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a 2D slice item with full processing.

        Returns dict with:
        - 'image': np.ndarray or Tensor of shape (C, H, W).
        - 'segmentations' (if enabled): segmentation masks with depth dimension removed.
        - 'image_labels': dict of annotator -> label tensor.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        result = self._get_raw_item(index)

        img = result['image']
        if isinstance(img, np.ndarray):
            img = self._preprocess_image_array(img)
        annotations = result['annotations']
        resource: SlicedVolumeResource = result['resource']

        # Process segmentations
        # FIXME: This currently re-loads the slice data for each segmentation annotation, which is inefficient. We should ideally load the slice once and reuse it for all segmentations. This may require refactoring how annotations are processed to avoid redundant data loading.
        if self.return_segmentations:
            seg_anns = AnnotationProcessor.filter_annotations(
                annotations, type='segmentation', scope='all'
            )
            segmentations, seg_labels, _ = self.annotation_processor.load_segmentations(seg_anns)

            # Slice segmentations along the same axis as the image
            # segmentations[author] shape: (#instances, D, H, W)
            slice_idx = resource.slice_index
            slice_axis_idx_std = resource.slice_axis_idx_std
            # resource normalized is always (D, C, H, W)
            map_slice_axis_idx_std = {
                0: 1,
                2: 2,
                3: 3,
            }
            seg_slice_axis_idx = map_slice_axis_idx_std[slice_axis_idx_std]
            sliced_segs: dict[str, np.ndarray] = {}
            for author, seg_array in segmentations.items():
                # seg_array shape: (#instances, D, H, W)
                # Select the slice: (#instances, DIM1, DIM2) where DIM1 and DIM2 depend on the slice axis
                sliced_segs[author] = np.take(seg_array, slice_idx, axis=seg_slice_axis_idx)
                # Add a dummy dimension for consistency with pipeline: (#instances, 1, DIM1, DIM2)
                sliced_segs[author] = np.expand_dims(sliced_segs[author], axis=1)

            # Apply albumentations if present
            if self.alb_transform:
                aug_result = self.apply_alb_transform(img, sliced_segs)
                img = aug_result['image']
                result['image'] = img
                sliced_segs = aug_result['segmentations']

            segmentations_processed, seg_labels_out = self._process_segmentations(sliced_segs, seg_labels,
                                                                                  output_shape=img.shape[1:])
            # remove temporary dummy dimension: (#instances, 1, DIM1, DIM2) -> (#instances, DIM1, DIM2)
            if isinstance(segmentations_processed, (Tensor, np.ndarray)):
                segmentations_processed = segmentations_processed.squeeze(1)
            elif isinstance(segmentations_processed, dict):
                for author in segmentations_processed:
                    if isinstance(segmentations_processed[author], (Tensor, np.ndarray)):
                        segmentations_processed[author] = segmentations_processed[author].squeeze(1)

            result['segmentations'] = segmentations_processed
            if seg_labels_out:
                result['seg_labels'] = seg_labels_out

        # Process image-level labels
        result['image_labels'] = self._extract_image_labels(annotations)

        # # Squeeze depth=1 from image: (C, 1, H, W) -> (C, H, W)
        # img = result['image']
        # _LOGGER.debug(f"Final image shape before squeezing depth: {img.shape}")
        # if isinstance(img, (np.ndarray, Tensor)) and img.ndim == 4 and img.shape[1] == 1:
        #     if isinstance(img, np.ndarray):
        #         img = img.squeeze(axis=1)
        #     else:
        #         img = img.squeeze(1)
        # result['image'] = img

        return result

    @override
    def apply_alb_transform(
        self,
        img: np.ndarray,
        segmentations: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Apply 2D albumentations transform to a single-slice image and masks.

        Uses the same approach as ImageDataset: treats the data as 2D.

        Args:
            img: Image array of shape (C, 1, H, W) or (C, H, W).
            segmentations: Dict of author -> mask arrays of shape (#instances, 1, H, W) or (#instances, H, W).

        Returns:
            Dict with transformed 'image' and 'segmentations'.
        """
        if self.alb_transform is None:
            raise ValueError("alb_transform is not set")

        # Squeeze depth=1 if present
        if img.ndim == 4:
            if img.shape[1] != 1:
                raise ValueError(f"Expected depth=1, got shape {img.shape}")
            img = img.squeeze(1)  # (C, 1, H, W) -> (C, H, W)
        elif img.ndim != 3:
            raise ValueError(f"Expected 3D or 4D image array, got shape {img.shape}")

        # Transpose to (H, W, C) for albumentations
        img = np.transpose(img, (1, 2, 0))

        replay_alb_transf = albumentations.ReplayCompose([self.alb_transform])

        aug_data = replay_alb_transf(image=img)
        replay_data = aug_data['replay']
        aug_img = aug_data['image']

        aug_segmentations: dict[str, np.ndarray] = {}
        for author, segs in segmentations.items():
            had_depth = False
            if segs.ndim == 4 and segs.shape[1] == 1:
                had_depth = True
                segs = segs.squeeze(1)  # (#instances, 1, H, W) -> (#instances, H, W)
            aug_segs = replay_alb_transf.replay(replay_data, masks=segs)['masks']
            if had_depth:
                aug_segs = aug_segs[:, np.newaxis, :, :]
            aug_segmentations[author] = aug_segs

        # Transpose back to (C, H, W)
        if isinstance(aug_img, np.ndarray):
            aug_img = np.transpose(aug_img, (2, 0, 1))
        elif isinstance(aug_img, torch.Tensor):
            if aug_img.shape[0] == img.shape[-1]:
                pass  # already (C, H, W)
            else:
                aug_img = aug_img.permute(2, 0, 1)

        # Add depth=1 back: (C, 1, H, W)
        aug_img = aug_img[:, np.newaxis, :, :]

        return {
            'image': aug_img,
            'segmentations': aug_segmentations,
        }

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"SlicedVolumeDataset (axis={self._slice_axis})\n{base}"
