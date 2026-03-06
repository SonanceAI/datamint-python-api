"""
SlicedVolumeDataset - 2D dataset created by slicing a VolumeDataset along an axis.

Provides a way to iterate over individual 2D slices from 3D volume data,
enabling training of 2D models on volumetric medical imaging data.
"""
import gzip
import hashlib
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

# Cache key for sliced segmentation numpy arrays
_SEG_SLICE_CACHEKEY = "seg_slice_array"


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
        seg_anns = [ann for ann in annotations if ann.annotation_type == 'segmentation']
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
        ann: 'Annotation',
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
