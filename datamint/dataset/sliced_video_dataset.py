"""
SlicedVideoDataset - 2D dataset created by iterating over frames of a VideoDataset.

Provides a way to iterate over individual 2D frames from video data,
enabling training of 2D models on temporal medical imaging data.
"""
from __future__ import annotations
import hashlib
import logging
from typing import Any, TYPE_CHECKING
from typing_extensions import override
from collections.abc import Sequence

import numpy as np
import torch
from torch import Tensor
import albumentations

from .base import DatamintBaseDataset
from .annotation_processor import AnnotationProcessor
from datamint.entities.cache_manager import CacheManager

if TYPE_CHECKING:
    from datamint.entities import Annotation
    from datamint.entities.sliced_video_resource import SlicedVideoResource

_LOGGER = logging.getLogger(__name__)

# Cache key for sliced segmentation numpy arrays
_SEG_FRAME_CACHEKEY = "seg_frame_array"


class SlicedVideoDataset(DatamintBaseDataset):
    """2D dataset created by iterating over frames of a video.

    Each item corresponds to a single frame from a video.
    The ``__getitem__`` returns arrays with shape ``(C, H, W)`` for images
    and ``(num_instances, H, W)`` or ``(num_labels+1, H, W)`` for segmentations.

    Can be instantiated directly with all the same parameters as
    :class:`DatamintBaseDataset`, or created from an already-loaded dataset
    via the :meth:`from_dataset` factory classmethod (which avoids additional
    server calls).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # --- Segmentation frame cache ---
        self._seg_frame_cache = CacheManager(
            'sliced_video_segmentations',
            enable_memory_cache=True,
            memory_cache_maxsize=8,
        )

        # --- Build per-frame resources ---
        frame_cache = CacheManager(
            'sliced_video_frames',
            enable_memory_cache=True,
            memory_cache_maxsize=2,
        )
        expanded_resources, expanded_annotations = self._expand_resources(
            self.resources,
            self.resource_annotations,
            frame_cache,
        )
        self.resources = expanded_resources  # type: ignore[assignment]
        self.resource_annotations = expanded_annotations

    @classmethod
    def from_dataset(
        cls,
        parent_dataset: DatamintBaseDataset,
    ) -> SlicedVideoDataset:
        """Create a SlicedVideoDataset from an existing dataset without additional server calls.

        Copies all configuration, label mappings, and already-loaded resources
        from ``parent_dataset``, then expands them into per-frame proxy resources.

        Args:
            parent_dataset: The source dataset (e.g. VideoDataset).

        Returns:
            A new :class:`SlicedVideoDataset` instance.
        """
        instance: SlicedVideoDataset = cls.__new__(cls)

        instance.project = parent_dataset.project

        # Copy configuration from parent
        instance.return_metainfo = parent_dataset.return_metainfo
        instance.return_segmentations = parent_dataset.return_segmentations
        instance.return_as_semantic_segmentation = parent_dataset.return_as_semantic_segmentation
        instance.semantic_seg_merge_strategy = parent_dataset.semantic_seg_merge_strategy
        instance.include_unannotated = parent_dataset.include_unannotated

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

        # --- Segmentation frame cache ---
        instance._seg_frame_cache = CacheManager(
            'sliced_video_segmentations',
            enable_memory_cache=True,
            memory_cache_maxsize=8,
        )

        # --- Build per-frame resources ---
        frame_cache = CacheManager(
            'sliced_video_frames',
            enable_memory_cache=True,
            memory_cache_maxsize=2,
        )
        expanded_resources, expanded_annotations = instance._expand_resources(
            parent_dataset.resources,
            parent_dataset.resource_annotations,
            frame_cache,
        )
        instance.resources = expanded_resources  # type: ignore[assignment]
        instance.resource_annotations = expanded_annotations

        return instance

    def _expand_resources(
        self,
        resources: Sequence,
        resource_annotations: Sequence[Sequence['Annotation']],
        frame_cache: CacheManager,
    ) -> tuple[list['SlicedVideoResource'], list[Sequence['Annotation']]]:
        """Expand video resources into per-frame proxy resources.

        Args:
            resources: Original video resources.
            resource_annotations: Parallel annotation sequences.
            frame_cache: Shared cache for decoded frames.

        Returns:
            Tuple of (frame_resources, frame_annotations).
        """
        from datamint.entities.sliced_video_resource import SlicedVideoResource

        frame_resources: list[SlicedVideoResource] = []
        frame_annotations: list[Sequence[Annotation]] = []

        for i, r in enumerate(resources):
            anns = resource_annotations[i]
            per_frame = SlicedVideoResource.slice_over(r, frame_cache)
            frame_resources.extend(per_frame)
            frame_annotations.extend(anns for _ in per_frame)

        return frame_resources, frame_annotations

    def _load_frame_segmentations(
        self,
        annotations: Sequence['Annotation'],
        resource: 'SlicedVideoResource',
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list]]:
        """Load segmentations for a specific video frame, with caching.

        Args:
            annotations: Segmentation annotations for this resource.
            resource: The SlicedVideoResource identifying the frame.

        Returns:
            Tuple of (frame_segs, seg_labels, seg_metainfos):
                - frame_segs: dict[author -> np.ndarray (#instances, 1, H, W)]
                - seg_labels: dict[author -> np.ndarray of int codes]
                - seg_metainfos: dict[author -> list]
        """
        seg_anns = [ann for ann in annotations if ann.annotation_type == 'segmentation']
        if not seg_anns:
            return {}, {}, {}

        image_seg_anns = [a for a in seg_anns if a.scope == 'image']
        frame_seg_anns = [a for a in seg_anns if a.scope == 'frame']

        uniq_authors = set(
            self.annotation_processor.get_author(a) for a in seg_anns
        )
        segmentations: dict[str, list[np.ndarray]] = {a: [] for a in uniq_authors}
        seg_labels: dict[str, list[int]] = {a: [] for a in uniq_authors}
        seg_metainfos: dict[str, list] = {a: [] for a in uniq_authors}

        # --- Image-scoped segmentations (full video masks) ---
        for ann in image_seg_anns:
            author = self.annotation_processor.get_author(ann)
            seg_code = self.annotation_processor.resolve_seg_code(ann.identifier)

            frame_seg = self._fetch_frame_seg_annotation(ann, resource)
            segmentations[author].append(frame_seg)
            seg_labels[author].append(seg_code)
            seg_metainfos[author].append(ann)

        # --- Frame-scoped segmentations ---
        if frame_seg_anns:
            frame_groups = self.annotation_processor.group_annotations(
                frame_seg_anns, by_author=True, by_identifier=True
            )
            for (author, identifier), fr_anns in frame_groups.items():
                seg_code = self.annotation_processor.resolve_seg_code(identifier)

                frame_seg = self._fetch_frame_seg_group(fr_anns, resource)
                if frame_seg is None:
                    continue
                segmentations[author].append(frame_seg)
                seg_labels[author].append(seg_code)
                seg_metainfos[author].append(fr_anns)

        # Stack per-author and add dummy depth dim
        final_segmentations: dict[str, np.ndarray] = {}
        final_seg_labels: dict[str, np.ndarray] = {}
        for author in segmentations:
            if segmentations[author]:
                stacked = np.stack(segmentations[author], axis=0)  # (#instances, H, W)
                stacked = np.expand_dims(stacked, axis=1)  # (#instances, 1, H, W)
                final_segmentations[author] = stacked
                final_seg_labels[author] = np.array(seg_labels[author], dtype=np.int32)

        return final_segmentations, final_seg_labels, seg_metainfos

    def _fetch_frame_seg_annotation(
        self,
        ann: 'Annotation',
        resource: 'SlicedVideoResource',
    ) -> np.ndarray:
        """Load an image-scoped segmentation and extract the frame, with caching.

        Args:
            ann: Image-scoped segmentation annotation.
            resource: Frame proxy identifying the frame index.

        Returns:
            2D boolean segmentation array of shape ``(H, W)``.
        """
        cache_entity_id = f"{ann.id}:frame{resource.frame_index}"
        version_info = {
            'created_at': ann.created_at,
            'deleted_at': ann.deleted_at,
            'associated_file': ann.associated_file,
        }

        cached = self._seg_frame_cache.get(cache_entity_id, _SEG_FRAME_CACHEKEY, version_info)
        if cached is not None:
            return cached

        # Load the full segmentation (N, H, W) and extract the frame
        full_seg = self.annotation_processor.load_segmentation_data(ann)
        # full_seg shape: (N, H, W) where N = number of frames
        frame_seg = full_seg[resource.frame_index]  # (H, W)
        frame_seg = np.ascontiguousarray(frame_seg)

        self._seg_frame_cache.set(cache_entity_id, _SEG_FRAME_CACHEKEY, frame_seg, version_info)
        return frame_seg

    def _fetch_frame_seg_group(
        self,
        fr_anns: list['Annotation'],
        resource: 'SlicedVideoResource',
    ) -> np.ndarray | None:
        """Collate frame-level segmentation annotations and extract the frame.

        Frame-level annotations each cover a single frame. Only the annotation
        matching the requested frame index contributes; others are ignored.

        Args:
            fr_anns: Frame-scoped annotations sharing the same author+identifier.
            resource: Frame proxy identifying the frame index.

        Returns:
            2D boolean array of shape ``(H, W)``, or None if no annotation matches.
        """
        ann_ids_str = ','.join(sorted(a.id or '' for a in fr_anns))
        group_hash = hashlib.sha256(ann_ids_str.encode()).hexdigest()[:16]
        cache_entity_id = f"frame_seg:{group_hash}:frame{resource.frame_index}"
        version_info = {
            'ann_ids_with_versions': [
                (a.id or '', a.associated_file, a.deleted_at)
                for a in sorted(fr_anns, key=lambda a: a.id or '')
            ],
        }

        cached = self._seg_frame_cache.get(cache_entity_id, _SEG_FRAME_CACHEKEY, version_info)
        if cached is not None:
            return cached

        # Find annotations matching the requested frame index
        matching = [a for a in fr_anns if a.frame_index == resource.frame_index]
        if not matching:
            # No annotation covers this frame → all zeros
            sample_seg = self.annotation_processor.load_segmentation_data(fr_anns[0])
            frame_seg = np.zeros(sample_seg.shape[1:], dtype=bool)  # (H, W)
            self._seg_frame_cache.set(cache_entity_id, _SEG_FRAME_CACHEKEY, frame_seg, version_info)
            return frame_seg

        # Union of all matching annotations at this frame index
        frame_seg: np.ndarray | None = None
        for ann in matching:
            seg = self.annotation_processor.load_segmentation_data(ann)  # (1, H, W)
            mask = seg[0]  # (H, W)
            frame_seg = mask if frame_seg is None else (frame_seg | mask)
        frame_seg = np.ascontiguousarray(frame_seg)

        self._seg_frame_cache.set(cache_entity_id, _SEG_FRAME_CACHEKEY, frame_seg, version_info)
        return frame_seg

    @override
    def _get_raw_item(self, index: int) -> dict[str, Any]:
        """Load a single video frame and its annotations.

        Returns dict with:
        - ``'image'``: np.ndarray of shape ``(C, H, W)``.
        - ``'annotations'``: Sequence of Annotation objects.
        - ``'resource'``: The SlicedVideoResource proxy.
        """
        resource: SlicedVideoResource = self.resources[index]  # type: ignore[assignment]
        img = resource.fetch_frame_data()  # (C, H, W)

        anns = self.resource_annotations[index]

        return {
            'image': img,
            'annotations': anns,
            'resource': resource,
        }

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        """Get a single frame item with full processing.

        Returns dict with:
        - ``'image'``: np.ndarray or Tensor of shape ``(C, H, W)``.
        - ``'segmentations'`` (if enabled): segmentation masks of shape ``(num_instances, H, W)`` or ``(num_labels+1, H, W)``.
        - ``'image_labels'``: dict of annotator -> label tensor.
        """
        if index >= len(self):
            raise IndexError(f"Index {index} out of bounds for dataset of size {len(self)}")

        result = self._get_raw_item(index)

        img = result['image']
        _LOGGER.debug(f'>>>>1 {img.shape=}')
        if isinstance(img, np.ndarray):
            img = self._preprocess_image_array(img)
        annotations = result['annotations']
        resource: SlicedVideoResource = result['resource']

        # Process segmentations
        if self.return_segmentations:
            seg_anns = AnnotationProcessor.filter_annotations(
                annotations, type='segmentation', scope='all'
            )
            frame_segs, seg_labels, _ = self._load_frame_segmentations(seg_anns, resource)

            # Apply albumentations if present
            if self.alb_transform:
                aug_result = self.apply_alb_transform(img, frame_segs)
                img = aug_result['image']
                result['image'] = img
                frame_segs = aug_result['segmentations']

            segmentations_processed, seg_labels_out = self._process_segmentations(
                frame_segs, seg_labels, output_shape=img.shape[1:],
            )
            # Remove dummy depth dimension: (#instances, 1, H, W) -> (#instances, H, W)
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

        return result

    @override
    def apply_alb_transform(
        self,
        img: np.ndarray,
        segmentations: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Apply 2D albumentations transform to a single frame and masks.

        Args:
            img: Image array of shape ``(C, H, W)``.
            segmentations: Dict of author -> mask arrays of shape
                ``(#instances, 1, H, W)`` or ``(#instances, H, W)``.

        Returns:
            Dict with transformed ``'image'`` and ``'segmentations'``.
        """
        if self.alb_transform is None:
            raise ValueError("alb_transform is not set")

        if img.ndim != 3:
            raise ValueError(f"Expected 3D image array (C, H, W), got shape {img.shape}")

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

        return {
            'image': aug_img,
            'segmentations': aug_segmentations,
        }

    def __repr__(self) -> str:
        base = super().__repr__()
        return f"SlicedVideoDataset\n{base}"
