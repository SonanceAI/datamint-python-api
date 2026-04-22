"""
SlicedVolumeDataset - 2D dataset created by slicing a VolumeDataset along an axis.

Provides a way to iterate over individual 2D slices from 3D volume data,
enabling training of 2D models on volumetric medical imaging data.
"""
from __future__ import annotations
import hashlib
from typing import Any, TYPE_CHECKING, cast
from typing_extensions import override
from collections.abc import Sequence
import numpy as np
import torch
from torch import Tensor
import albumentations

from .base import DatamintBaseDataset
from .annotation_processor import AnnotationProcessor

import logging

from datamint.entities.cache_manager import CacheManager
if TYPE_CHECKING:
    from datamint.entities import Annotation, Resource
    from datamint.entities.sliced_resource import SlicedVolumeResource
    from medimgkit import ViewPlane

_LOGGER = logging.getLogger(__name__)

# Cache key for sliced segmentation numpy arrays
_SEG_SLICE_CACHEKEY = "seg_slice_array"


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

    Can be instantiated directly with all the same parameters as
    :class:`DatamintBaseDataset` plus ``slice_axis``, or created from an
    already-loaded dataset via the :meth:`from_dataset` factory classmethod
    (which avoids additional server calls).

    Args:
        project: Project name, Project object, or None. Mutually exclusive with resources.
        resources: List of Resource objects, or None. Mutually exclusive with project.
        slice_axis: Slice orientation. One of ``'axial'`` (depth), ``'coronal'``
            (height), ``'sagittal'`` (width), or an integer axis index (0--2).
        See :class:`DatamintBaseDataset` for all remaining parameters.
    """

    def __init__(
        self,
        *args,
        slice_axis: ViewPlane | int = 'axial',
        **kwargs,
    ):
        self._slice_axis = cast('ViewPlane', self._validate_slice_axis(slice_axis))

        super().__init__(
            *args,
            **kwargs,
        )

        # --- Segmentation slice cache ---
        self._seg_slice_cache = CacheManager(
            'sliced_segmentations',
            enable_memory_cache=True,
            memory_cache_maxsize=8,
        )

        # --- Build sliced resources ---
        volume_cache = CacheManager(
            'sliced_volumes',
            enable_memory_cache=True,
            memory_cache_maxsize=2,
        )
        expanded_resources, expanded_annotations = self._expand_resources(
            self.resources,
            self.resource_annotations,
            volume_cache,
        )
        self.resources = expanded_resources  # type: ignore[assignment]
        self.resource_annotations = expanded_annotations

    @classmethod
    def from_dataset(
        cls,
        parent_dataset: DatamintBaseDataset,
        slice_axis: 'ViewPlane | int' = 'axial',
    ) -> 'SlicedVolumeDataset':
        """Create a SlicedVolumeDataset from an existing dataset without additional server calls.

        Copies all configuration, label mappings, and already-loaded resources
        from ``parent_dataset``, then expands them into per-slice proxy resources.
        Use this factory when you already have a loaded dataset and want to obtain
        2D slices without triggering new API requests.

        Args:
            parent_dataset: The source :class:`DatamintBaseDataset` (e.g. VolumeDataset)
                providing resources, annotations, and configuration.
            slice_axis: Slice orientation. One of ``'axial'`` (depth), ``'coronal'``
                (height), ``'sagittal'`` (width), or an integer axis index (0--2).

        Returns:
            A new :class:`SlicedVolumeDataset` instance.
        """
        parent_dataset._prepare()
        instance: SlicedVolumeDataset = cls.__new__(cls)

        instance._slice_axis = cast('ViewPlane', cls._validate_slice_axis(slice_axis))

        instance.project = parent_dataset.project
        instance._api = parent_dataset._api

        # Copy configuration from parent
        instance.return_metainfo = parent_dataset.return_metainfo
        instance.return_segmentations = parent_dataset.return_segmentations
        instance.return_as_semantic_segmentation = parent_dataset.return_as_semantic_segmentation
        instance.semantic_seg_merge_strategy = parent_dataset.semantic_seg_merge_strategy
        instance.include_unannotated = parent_dataset.include_unannotated
        instance.allow_external_annotations = parent_dataset.allow_external_annotations
        instance.image_labels_merge_strategy = parent_dataset.image_labels_merge_strategy
        instance.image_categories_merge_strategy = parent_dataset.image_categories_merge_strategy
        instance.split_name = parent_dataset.split_name
        instance.split_source = parent_dataset.split_source
        instance.split_as_of_timestamp = parent_dataset.split_as_of_timestamp

        # Transforms
        instance.alb_transform = parent_dataset.alb_transform

        # Filtering (already applied on parent's annotations)
        instance.include_annotators = parent_dataset.include_annotators
        instance.exclude_annotators = parent_dataset.exclude_annotators
        instance.include_segmentation_names = parent_dataset.include_segmentation_names
        instance.exclude_segmentation_names = parent_dataset.exclude_segmentation_names
        instance.include_image_label_names = parent_dataset.include_image_label_names
        instance.exclude_image_label_names = parent_dataset.exclude_image_label_names
        instance.include_frame_label_names = parent_dataset.include_frame_label_names
        instance.exclude_frame_label_names = parent_dataset.exclude_frame_label_names

        # Copy label sets and processor from parent
        instance.annotation_processor = parent_dataset.annotation_processor
        instance.frame_lsets = parent_dataset.frame_lsets
        instance.frame_lcodes = parent_dataset.frame_lcodes
        instance.image_lsets = parent_dataset.image_lsets
        instance.image_lcodes = parent_dataset.image_lcodes
        instance.seglabel_list = parent_dataset.seglabel_list
        instance.seglabel2code = parent_dataset.seglabel2code

        # Internal state
        instance._logged_uint16_conversion = False
        instance._is_prepared = True

        # --- Segmentation slice cache ---
        instance._seg_slice_cache = CacheManager(
            'sliced_segmentations',
            enable_memory_cache=True,
            memory_cache_maxsize=8,
        )

        # --- Build sliced resources ---
        volume_cache = CacheManager(
            'sliced_volumes',
            enable_memory_cache=True,
            memory_cache_maxsize=2,
        )
        expanded_resources, expanded_annotations = instance._expand_resources(
            parent_dataset.resources,
            parent_dataset.resource_annotations,
            volume_cache,
        )
        instance.resources = expanded_resources  # type: ignore[assignment]
        instance.resource_annotations = expanded_annotations

        return instance

    @staticmethod
    def _validate_slice_axis(slice_axis: 'ViewPlane | int') -> 'ViewPlane':
        if isinstance(slice_axis, str):
            valid_slice_axis = ['axial', 'coronal', 'sagittal']
            if slice_axis not in valid_slice_axis:
                raise ValueError(
                    f"Unknown axis '{slice_axis}'. "
                    f"Must be one of {valid_slice_axis} or an int 0-2."
                )
            return slice_axis

        if not (0 <= slice_axis <= 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {slice_axis}")

        axis_names: dict[int, ViewPlane] = {0: 'axial', 1: 'coronal', 2: 'sagittal'}
        return axis_names[slice_axis]

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
        from datamint.entities.sliced_resource import SlicedVolumeResource

        sliced_resources: list[SlicedVolumeResource] = []
        sliced_annotations: list[Sequence['Annotation']] = []

        requires_download = any(not r.is_cached() for r in resources)
        iterator = enumerate(resources)
        if requires_download:
            from tqdm.auto import tqdm
            _LOGGER.warning(
                "Some resources are not cached locally and will be downloaded during slicing. "
                "This may take time and bandwidth, especially for large volumes. "
                "Consider pre-caching resources if this is an issue.")
            iterator = tqdm(iterator, total=len(resources), desc=f"Expanding to '{self._slice_axis}' slices")

        for i, r in iterator:
            anns = resource_annotations[i]
            per_slice = SlicedVolumeResource.slice_over(r, self._slice_axis, volume_cache)
            sliced_resources.extend(per_slice)
            sliced_annotations.extend(anns for _ in per_slice)

        return sliced_resources, sliced_annotations

    # --- Axis mapping for segmentation slicing ---
    # Volume is normalized to (D, C, H, W). load_segmentation_data returns (D, H, W).
    # Axis 1 is the channel axis and is never sliced — the mapping is simply:
    #   std_axis == 0  ->  seg axis 0  (depth)
    #   std_axis == 2  ->  seg axis 1  (height)
    #   std_axis == 3  ->  seg axis 2  (width)
    @staticmethod
    def _std_axis_to_seg_axis(slice_axis_idx_std: int) -> int:
        """Convert a standardised 4D volume axis index to the matching 3D seg array axis."""
        if slice_axis_idx_std == 0:
            return 0
        if slice_axis_idx_std in (2, 3):
            return slice_axis_idx_std - 1
        raise ValueError(
            f"Cannot slice along channel axis (std axis 1). Got slice_axis_idx_std={slice_axis_idx_std}"
        )

    def _load_sliced_segmentations(
        self,
        annotations: Sequence['Annotation'],
        resource: SlicedVolumeResource,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list]]:
        """Load segmentations already sliced for a specific 2D slice, with caching.

        Instead of loading entire 3D segmentation volumes and then slicing,
        this method caches sliced 2D segmentations per annotation to avoid
        redundant volume parsing on repeated access.

        Args:
            annotations: Segmentation annotations for this resource.
            resource: The SlicedVolumeResource identifying the slice.

        Returns:
            Tuple of (sliced_segs, seg_labels, seg_metainfos):
                - sliced_segs: dict[author -> np.ndarray (#instances, 1, DIM1, DIM2)]
                - seg_labels: dict[author -> np.ndarray of int codes]
                - seg_metainfos: dict[author -> list]
        """
        seg_anns = [ann for ann in annotations if ann.is_segmentation()]
        if not seg_anns:
            return {}, {}, {}

        seg_slice_axis = self._std_axis_to_seg_axis(resource.slice_axis_idx_std)

        image_seg_anns = [a for a in seg_anns if a.scope == 'image']
        frame_seg_anns = [a for a in seg_anns if a.scope == 'frame']

        uniq_authors = set(
            self.annotation_processor.get_author(a) for a in seg_anns
        )
        segmentations: dict[str, list[np.ndarray]] = {a: [] for a in uniq_authors}
        seg_labels: dict[str, list[int]] = {a: [] for a in uniq_authors}
        seg_metainfos: dict[str, list] = {a: [] for a in uniq_authors}

        # --- Image-scoped segmentations ---
        for ann in image_seg_anns:
            author = self.annotation_processor.get_author(ann)
            seg_code = self.annotation_processor.resolve_seg_code(ann.identifier)

            sliced_seg = self._fetch_sliced_seg_annotation(ann, resource, seg_slice_axis)
            # sliced_seg shape: (DIM1, DIM2)
            segmentations[author].append(sliced_seg)
            seg_labels[author].append(seg_code)
            seg_metainfos[author].append(ann)

        # --- Frame-scoped segmentations ---
        if frame_seg_anns:
            frame_groups = self.annotation_processor.group_annotations(
                frame_seg_anns, by_author=True, by_identifier=True
            )
            for (author, identifier), fr_anns in frame_groups.items():
                seg_code = self.annotation_processor.resolve_seg_code(identifier)

                sliced_seg = self._fetch_sliced_frame_seg_group(
                    fr_anns, resource, seg_slice_axis
                )
                if sliced_seg is None:
                    continue
                segmentations[author].append(sliced_seg)
                seg_labels[author].append(seg_code)
                seg_metainfos[author].append(fr_anns)

        # Stack per-author and add dummy depth dim
        final_segmentations: dict[str, np.ndarray] = {}
        final_seg_labels: dict[str, np.ndarray] = {}
        for author in segmentations:
            if segmentations[author]:
                stacked = np.stack(segmentations[author], axis=0)  # (#instances, DIM1, DIM2)
                stacked = np.expand_dims(stacked, axis=1)  # (#instances, 1, DIM1, DIM2)
                final_segmentations[author] = stacked
                final_seg_labels[author] = np.array(seg_labels[author], dtype=np.int32)

        return final_segmentations, final_seg_labels, seg_metainfos

    def _fetch_sliced_seg_annotation(
        self,
        ann: Annotation,
        resource: SlicedVolumeResource,
        seg_slice_axis: int,
    ) -> np.ndarray:
        """Load a single image-scoped segmentation, slice it, and cache the 2D result.

        On the first call the full 3D segmentation is loaded via
        ``load_segmentation_data`` (raw bytes are already cached by the
        annotation entity). The requested 2D slice is extracted, cached in
        ``_seg_slice_cache``, and returned. Subsequent calls for the same
        annotation + slice return the cached array directly, avoiding the
        expensive volume parsing.

        Args:
            ann: Image-scoped segmentation annotation.
            resource: Slice proxy identifying axis and index.
            seg_slice_axis: Axis index in the (D, H, W) segmentation array.

        Returns:
            2D boolean segmentation array of shape (DIM1, DIM2).
        """
        cache_entity_id = f"{ann.id}:axis{resource.slice_axis}:slice{resource.slice_index}"
        version_info = {
            'created_at': ann.created_at,
            'deleted_at': ann.deleted_at,
            'associated_file': ann.associated_file,
        }

        cached = self._seg_slice_cache.get(cache_entity_id, _SEG_SLICE_CACHEKEY, version_info)
        if cached is not None:
            return cached

        # Load the full 3D segmentation volume (raw bytes are cached by ann.fetch_file_data)
        full_seg = self.annotation_processor.load_segmentation_data(ann)
        # full_seg shape: (D, H, W)
        sliced = np.take(full_seg, resource.slice_index, axis=seg_slice_axis)
        sliced = np.ascontiguousarray(sliced)

        self._seg_slice_cache.set(cache_entity_id, _SEG_SLICE_CACHEKEY, sliced, version_info)
        return sliced

    def _fetch_sliced_frame_seg_group(
        self,
        fr_anns: list['Annotation'],
        resource: SlicedVolumeResource,
        seg_slice_axis: int,
    ) -> np.ndarray | None:
        """Collate frame-level segmentation annotations, slice, and cache the 2D result.

        Frame-level annotations each cover a single frame (depth index). They
        are first assembled into a full (D, H, W) volume via
        ``collate_frame_segmentations``, then sliced and cached. Subsequent
        calls for the same group + slice return the cached array.

        Args:
            fr_anns: Frame-scoped annotations sharing the same author+identifier.
            resource: Slice proxy identifying axis and index.
            seg_slice_axis: Axis index in the (D, H, W) segmentation array.

        Returns:
            2D boolean array of shape (DIM1, DIM2), or None if collation yields nothing.
        """
        # Build a stable cache key from a hash of sorted annotation IDs
        ann_ids_str = ','.join(sorted(a.id or '' for a in fr_anns))
        group_hash = hashlib.sha256(ann_ids_str.encode()).hexdigest()[:16]
        cache_entity_id = f"frame_seg:{group_hash}:axis{resource.slice_axis}:slice{resource.slice_index}"
        version_info = {
            'ann_ids_with_versions': [
                (a.id or '', a.associated_file, a.deleted_at)
                for a in sorted(fr_anns, key=lambda a: a.id or '')
            ],
        }

        cached = self._seg_slice_cache.get(cache_entity_id, _SEG_SLICE_CACHEKEY, version_info)
        if cached is not None:
            return cached

        # Fast path: when slicing along the depth axis, only the annotation
        # whose frame_index matches the requested slice contributes non-zero
        # data — load just that one instead of assembling the full volume.
        if seg_slice_axis == 0:
            matching = [a for a in fr_anns if a.frame_index == resource.slice_index]
            if not matching:
                # None of the frame annotations cover this depth slice → all zeros
                # We need the spatial shape; load just the first ann to obtain it.
                sample_seg = self.annotation_processor.load_segmentation_data(fr_anns[0])
                sliced = np.zeros(sample_seg.shape[1:], dtype=bool)  # (H, W)
                self._seg_slice_cache.set(cache_entity_id, _SEG_SLICE_CACHEKEY, sliced, version_info)
                return sliced

            # Union of all matching annotations at this depth index
            sliced: np.ndarray | None = None
            for ann in matching:
                seg = self.annotation_processor.load_segmentation_data(ann)  # (1, H, W)
                frame_mask = seg[0]  # (H, W)
                sliced = frame_mask if sliced is None else (sliced | frame_mask)
            sliced = np.ascontiguousarray(sliced)
            self._seg_slice_cache.set(cache_entity_id, _SEG_SLICE_CACHEKEY, sliced, version_info)
            return sliced

        # Non-depth axis: must assemble the full volume then slice along the chosen axis.
        stacked_seg, _ = self.annotation_processor.collate_frame_segmentations(fr_anns)
        if stacked_seg is None:
            return None
        # stacked_seg shape: (D, H, W)
        sliced = np.take(stacked_seg, resource.slice_index, axis=seg_slice_axis)
        sliced = np.ascontiguousarray(sliced)

        self._seg_slice_cache.set(cache_entity_id, _SEG_SLICE_CACHEKEY, sliced, version_info)
        return sliced

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
        if self.return_segmentations:
            seg_anns = AnnotationProcessor.filter_annotations(
                annotations, type='segmentation', scope='all'
            )
            sliced_segs, seg_labels, _ = self._load_sliced_segmentations(seg_anns, resource)

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
        orig_dim = img.ndim
        if orig_dim == 4:
            if img.shape[1] != 1:
                raise ValueError(f"Expected depth=1, got shape {img.shape}")
            img = img.squeeze(1)  # (C, 1, H, W) -> (C, H, W)
        elif orig_dim != 3:
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

        # Add depth=1 back: (C, 1, H, W) if original had it, else keep (C, H, W)
        if orig_dim == 4:
            aug_img = aug_img[:, np.newaxis, :, :]

        return {
            'image': aug_img,
            'segmentations': aug_segmentations,
        }

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"SlicedVolumeDataset (axis={self._slice_axis})\n{base}"
