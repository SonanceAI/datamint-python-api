"""
AnnotationProcessor - Handles segmentation and label processing.

This module provides annotation processing classes for different dataset types:
- BaseAnnotationProcessor: Generic processor with shared logic for all dataset types
- ImageAnnotationProcessor: Processor for simple 2D images (no frame/slot concept)
- SequenceAnnotationProcessor: Extended processor for multi-frame/multi-slot data (videos, volumes)

The class hierarchy ensures that the base class contains only generic logic that works
for any dataset type, while specialized logic is in subclasses.
"""
import logging
from typing import Literal, TYPE_CHECKING
from collections.abc import Iterable, Sequence
from collections import defaultdict

import numpy as np
import torch
from torch import Tensor
from typing_extensions import overload

from medimgkit.readers import read_array_normalized

if TYPE_CHECKING:
    from datamint.entities.annotations.annotation import Annotation

_LOGGER = logging.getLogger(__name__)

# Type alias for merge strategy
MergeStrategy = Literal['union', 'intersection', 'mode']


class AnnotationProcessor:
    """Base processor for annotations - contains only generic shared logic.

    This class provides generic annotation processing that works for any dataset type:
    - Loading segmentation data from annotations (raw load, no frame handling)
    - Generic merging strategies for semantic segmentations
    - Label name conversion utilities
    - Annotation filtering utilities

    Subclasses (ImageAnnotationProcessor, SequenceAnnotationProcessor) handle
    dataset-specific logic like frame/slot assignment and dimension handling.

    Args:
        seglabel2code: Mapping from label name to code.
        image_labels_set: List of image-level label names.
        image_lcodes: Mapping for image labels.
    """

    def __init__(
        self,
        seglabel2code: dict[str, int],
        image_labels_set: list[str],
        image_lcodes: dict[str, dict[str, int]],
    ):
        self.seglabel2code = seglabel2code
        self.image_labels_set = image_labels_set
        self.image_lcodes = image_lcodes

    def collate_frame_segmentations(self,
                                    fr_anns: Sequence['Annotation'],
                                    depth: int | None = None) -> tuple[np.ndarray | None, int]:
        stacked_seg = None
        seg_code = -1
        # sort frame annotations by frame index
        for ann in fr_anns:
            try:
                seg = self.load_segmentation_data(ann)
                seg_code_i = self.seglabel2code.get(ann.identifier, 0)
                _LOGGER.debug(f'Processing frame annotation {ann.id} at index {ann.frame_index} with shape {seg.shape}')
                if seg_code != -1 and seg_code != seg_code_i:
                    raise ValueError(f"Conflicting segmentation codes for frame annotations: "
                                     f"{seg_code} vs {seg_code_i}")
                seg_code = seg_code_i
                # seg shape: (1, H, W)
                seg = seg[0]  # -> (H, W)
                if stacked_seg is None:
                    if depth is None:
                        depth = ann.resource.get_depth()
                    stacked_seg = np.zeros((depth, *seg.shape), dtype=bool)
                if ann.frame_index is None:
                    raise ValueError(f"Frame-level annotation {ann.id} missing frame_index")
                stacked_seg[ann.frame_index] = seg
            except Exception as e:
                _LOGGER.error(f"Failed to load segmentation for annotation {ann.id}: {e}")
                raise
        return stacked_seg, seg_code

    def group_annotations(self,
                          annotations: Iterable['Annotation'],
                          by_author: bool = False,
                          by_identifier: bool = False,
                          ) -> dict[tuple, list['Annotation']]:
        """Group annotations by author and/or identifier.

        Args:
            annotations: Iterable of Annotation objects.
            by_author: If True, group by author.
            by_identifier: If True, group by identifier.

        Returns:
            Dict mapping grouping keys to lists of annotations.
        """

        if not by_author and not by_identifier:
            raise ValueError("At least one of grouping criteria must be True")

        seg_frame_anns_map = {}
        for ann in annotations:
            key_parts = []
            if by_author:
                author = ann.created_by or "unknown"
                key_parts.append(author)
            if by_identifier:
                identifier = ann.identifier
                key_parts.append(identifier)

            key = tuple(key_parts)
            if key not in seg_frame_anns_map:
                seg_frame_anns_map[key] = []
            seg_frame_anns_map[key].append(ann)

        return seg_frame_anns_map

    def load_image_segmentations(self,
                                 annotations: Iterable['Annotation']
                                 ) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
        """Load segmentations defined at image scope.
        Args:
            annotations: Iterable of Annotation objects (segmentation type).
        Returns:
            Tuple of (segmentations, seg_labels, seg_anns):
                - segmentations: dict[author -> list of mask arrays of shape (#slices, H, W)]
                - seg_labels: dict[author -> list of int codes]
                - seg_anns: dict[author -> list of Annotation objects]
        """

        segmentations = defaultdict(list)
        seg_labels = defaultdict(list)
        seg_anns = defaultdict(list)

        seg_image_annotations = [ann for ann in annotations
                                 if ann.scope == 'image' and ann.annotation_type == 'segmentation']
        for ann in seg_image_annotations:
            author = ann.created_by or ann.created_by_model or "unknown"

            try:
                seg = self.load_segmentation_data(ann)
                seg_code = self.seglabel2code.get(ann.identifier, 0)
                # seg shape: (#slices, H, W)
            except Exception as e:
                _LOGGER.error(f"Failed to load segmentation for annotation {ann.id}: {e}")
                raise

            if ann.frame_index is None:
                segmentations[author].append(seg)
            else:
                raise ValueError(f"unexpected scope/index combo for annotation {ann.id} "
                                 f"(scope={ann.scope}, frame_index={ann.frame_index})")
            seg_labels[author].append(seg_code)
            seg_anns[author].append(ann)

        return segmentations, seg_labels, seg_anns

    def load_frame_segmentations(
        self,
        annotations: Iterable['Annotation'],
    ) -> tuple[dict[str, list], dict[str, list], dict[str, list]]:
        """Load frame-level segmentations

        Args:
            annotations: Iterable of Annotation objects (segmentation type).
        Returns:
            Tuple of (segmentations, seg_labels, seg_metainfos):
                - segmentations: dict[author -> list of np.ndarray of shape (#num_instances, #frames, H, W)]
                - seg_labels: dict[author -> list of int codes]
                - seg_anns: dict[author -> list of list of Annotation objects]
        """
        annotations = [ann for ann in annotations
                       if ann.scope == 'frame' and ann.annotation_type == 'segmentation']
        seg_frame_anns_map = self.group_annotations(
            annotations,
            by_author=True,
            by_identifier=True,
        )

        segmentations = defaultdict(list)
        seg_labels = defaultdict(list)
        seg_anns = defaultdict(list)

        _LOGGER.debug(
            f"Found {len(seg_frame_anns_map)} unique (author, identifier) groups for frame-level segmentations")

        for (author, identifier), fr_anns in seg_frame_anns_map.items():
            stacked_seg, seg_code = self.collate_frame_segmentations(fr_anns)
            if stacked_seg is None:
                continue

            segmentations[author].append(stacked_seg)
            seg_labels[author].append(seg_code)
            seg_anns[author].append(fr_anns)

        return segmentations, seg_labels, seg_anns

    def _stack_segmentations(self,
                             segmentations: dict[str, list[np.ndarray]],
                             seg_labels: dict[str, list[int]]):
        # Stack per-author segmentations
        final_segmentations: dict[str, np.ndarray] = {}
        final_seg_labels: dict[str, np.ndarray] = {}
        for author in segmentations:
            _LOGGER.debug(f"Author {author} has {len(segmentations[author])} segmentations to stack")
            final_segmentations[author] = np.stack(segmentations[author], axis=0)  # (#num_instances, Z, H, W)
            final_seg_labels[author] = np.array(seg_labels[author], dtype=np.int32)

        return final_segmentations, final_seg_labels

    def load_segmentations(
        self,
        annotations: Iterable['Annotation']
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, list]]:
        """Load segmentations for multi-slot data (videos, volumes).

        Args:
            annotations: Iterable of Annotation objects (segmentation type).

        Returns:
            Tuple of (segmentations, seg_labels, seg_metainfos):
                - segmentations: dict[author -> np.ndarray of shape (#num_instances, depth or #slices or #frames, H, W)]
                - seg_labels: dict[author -> np.ndarray of #num_instances ints]
                - seg_metainfos: dict[author -> list of Annotation objects]
        """

        seg_annotations = [ann for ann in annotations if ann.annotation_type == 'segmentation']
        uniq_authors = set(ann.created_by or ann.created_by_model or "unknown" for ann in seg_annotations)
        segmentations: dict[str, list[np.ndarray]] = {a: []
                                                      for a in uniq_authors}  # tensors of shape (D, H, W)
        seg_labels: dict[str, list[int]] = {a: [] for a in uniq_authors}  # list of size=#num_instances
        seg_metainfos: dict[str, list] = {a: [] for a in uniq_authors}

        # group segmentations by author, identifier and filtered by scope
        fsegs, fseg_labels, fseg_anns = self.load_frame_segmentations(seg_annotations)
        for author in fsegs:
            segmentations[author].extend(fsegs[author])
            seg_labels[author].extend(fseg_labels[author])
            seg_metainfos[author].extend(fseg_anns[author])

        isegs, iseg_labels, iseg_anns = self.load_image_segmentations(seg_annotations)
        for author in isegs:
            segmentations[author].extend(isegs[author])
            seg_labels[author].extend(iseg_labels[author])
            seg_metainfos[author].extend(iseg_anns[author])

        final_segmentations, final_seg_labels = self._stack_segmentations(segmentations, seg_labels)

        assert len(final_segmentations) == len(final_seg_labels)
        for author in final_segmentations:
            assert final_segmentations[author].shape[0] == final_seg_labels[author].shape[0]

        return final_segmentations, final_seg_labels, seg_metainfos

    def load_segmentation_data(self, ann: 'Annotation',
                               auto_convert_gray: bool = True) -> np.ndarray:
        """Load segmentation data from an annotation.

        Args:
            ann: The annotation to load data from.
            auto_convert_gray: If True, convert multi-channel grayscale to single channel.

        Returns:
            np.ndarray: Binary segmentation array with shape (N, H, W).
                For image-level: N=#frames or #slices or depth
                For frame-level: N=1
        """
        if ann.type != 'segmentation':
            raise ValueError(f"Annotation {ann.id} is not a segmentation")

        ann_data_bytes = ann.fetch_file_data(use_cache=True, auto_convert=False)
        ann_data_array = read_array_normalized(ann_data_bytes)

        # Validate shape based on scope
        if len(ann_data_array.shape) != 4:
            raise ValueError(
                f"Segmentation annotation {ann.id} has invalid shape "
                f"{ann_data_array.shape}, expected 4D (N, C, H, W)"
            )

        if auto_convert_gray and np.allclose(ann_data_array[:, 0:1, :, :], ann_data_array[:, 1:3, :, :]):
            if ann_data_array.shape[1] == 4:
                _LOGGER.debug('RGBA detected. Ignoring alpha channel for annotation')
            ann_data_array = ann_data_array[:, 0:1, :, :]  # (N, 1, H, W)

        _LOGGER.debug(
            f"Loaded segmentation for annotation {ann.id} "
            f"with shape {ann_data_array.shape}"
        )

        # Validate and extract single channel
        if ann_data_array.shape[1] != 1:
            raise ValueError(f"Segmentation must have 1 channel, got shape {ann_data_array.shape}")
        ann_data_array = ann_data_array[:, 0, :, :]  # (N, H, W)
        return ann_data_array != 0  # binary mask

    def _merge_union(self, segmentations: dict[str, Tensor]) -> Tensor:
        """Union merge: pixel is labeled if ANY annotator labeled it."""
        new_segmentations = torch.zeros_like(list(segmentations.values())[0])
        for seg in segmentations.values():
            new_segmentations += seg
        return new_segmentations.bool()

    def _merge_intersection(self, segmentations: dict[str, Tensor]) -> Tensor:
        """Intersection merge: pixel is labeled if ALL annotators labeled it."""
        new_segmentations = torch.ones_like(list(segmentations.values())[0])
        for seg in segmentations.values():
            new_segmentations *= seg
        return new_segmentations.bool()

    def _merge_mode(self, segmentations: dict[str, Tensor]) -> Tensor:
        """Mode merge: pixel is labeled if majority of annotators labeled it."""
        new_segmentations = torch.zeros_like(list(segmentations.values())[0])
        for seg in segmentations.values():
            new_segmentations += seg
        new_segmentations = new_segmentations >= len(segmentations) / 2
        return new_segmentations

    def convert_image_labels(
        self,
        annotations: Sequence['Annotation'],
    ) -> dict[str, torch.Tensor]:
        """Convert image-level label annotations to one-hot tensors.

        Args:
            annotations: List of label annotations (image-scoped).

        Returns:
            Dict of annotator_id -> one-hot tensor of shape (num_labels,).
        """
        labels_ret_size = (len(self.image_labels_set),)
        label2code = self.image_lcodes.get('multilabel', {})

        labels_by_user: dict[str, torch.Tensor] = {}

        for ann in annotations:
            if ann.annotation_type != 'label':
                continue

            user_id = ann.created_by or "unknown"
            if user_id not in labels_by_user:
                labels_by_user[user_id] = torch.zeros(size=labels_ret_size, dtype=torch.int32)

            code = label2code.get(ann.identifier)
            if code is not None:
                labels_by_user[user_id][code] = 1

        return labels_by_user

    @overload
    def apply_merge_strategy(
        self,
        segmentations: dict[str, Tensor],
        strategy: MergeStrategy,
        output_shape: tuple[int, ...] | None = None,
    ) -> Tensor: ...

    @overload
    def apply_merge_strategy(
        self,
        segmentations: dict[str, np.ndarray],
        strategy: MergeStrategy,
        output_shape: tuple[int, ...] | None = None,
    ) -> np.ndarray: ...

    def apply_merge_strategy(
        self,
        segmentations: dict[str, Tensor] | dict[str, np.ndarray],
        strategy: MergeStrategy,
        output_shape: tuple[int, ...] | None = None,
    ) -> Tensor | np.ndarray:
        """Merge semantic segmentations from multiple annotators.

        Args:
            segmentations: Dict of author -> semantic segmentation tensor.
            output_shape: Shape for empty result if no segmentations are present.
            strategy: Merge strategy ('union', 'intersection', 'mode').

        Returns:
            Merged tensor if strategy is specified, otherwise original dict.
        """
        if len(segmentations) == 0:
            if output_shape is None:
                raise ValueError("output_shape must be provided when no segmentations are present")
            empty_segs = torch.zeros(output_shape, dtype=torch.get_default_dtype())
            empty_segs[0] = 1  # background
            return empty_segs

        if isinstance(next(iter(segmentations.values())), np.ndarray):
            with torch.no_grad():
                segmentations = {author: torch.from_numpy(seg) for author, seg in segmentations.items()}
                return self.apply_merge_strategy(segmentations, strategy, output_shape).numpy()

        _LOGGER.debug(
            f"Applying merge strategy '{strategy}' to {len(segmentations)} segmentations of type {type(next(iter(segmentations.values())))}")
        if strategy == 'union':
            merged = self._merge_union(segmentations)
        elif strategy == 'intersection':
            merged = self._merge_intersection(segmentations)
        elif strategy == 'mode':
            merged = self._merge_mode(segmentations)
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        return merged.to(torch.get_default_dtype())

    @overload
    def instance_to_semantic_segmentation(
        self,
        segmentations: None,
        seg_labels: Tensor | np.ndarray,
        num_labels: int
    ) -> None: ...

    @overload
    def instance_to_semantic_segmentation(
        self,
        segmentations: Tensor,
        seg_labels: Tensor,
        num_labels: int
    ) -> Tensor: ...

    @overload
    def instance_to_semantic_segmentation(
        self,
        segmentations: np.ndarray,
        seg_labels: np.ndarray,
        num_labels: int
    ) -> np.ndarray: ...

    def instance_to_semantic_segmentation(
        self,
        segmentations: Tensor | np.ndarray | None,
        seg_labels: Tensor | np.ndarray,
        num_labels: int
    ) -> Tensor | np.ndarray | None:
        """Convert instance segmentation to semantic segmentation for a sequence.

        Args:
            segmentations: Tensor/array of shape (num_instances, depth, H, W).
            seg_labels: Tensor/array of shape (num_instances,).

        Returns:
            If segmentations is a Sequence: Tensor/array of shape (num_labels+1, depth, H, W);
            If segmentations is None: None;
            If segmentations is a Tensor/array: Tensor/array of shape (num_labels+1, depth, H, W).
        """
        if segmentations is None:
            return None

        if isinstance(segmentations, np.ndarray):
            with torch.no_grad():
                segmentations = torch.from_numpy(segmentations)
                seg_labels = torch.from_numpy(seg_labels)
                return self.instance_to_semantic_segmentation(segmentations, seg_labels, num_labels).numpy()

        if len(segmentations) != len(seg_labels):
            raise ValueError("segmentations and seg_labels must have the same length")

        if len(segmentations) == 0:
            return torch.zeros((num_labels + 1, 0, 0, 0), dtype=torch.float32)

        depth, h, w = segmentations[0].shape

        semantic_seg = torch.zeros((num_labels + 1, depth, h, w), dtype=torch.uint8)

        for instance_idx in range(len(segmentations)):
            instance_seg = segmentations[instance_idx]
            instance_label = seg_labels[instance_idx].item()
            if instance_label == 0:
                raise ValueError(f"Instance {instance_idx} has label code 0 (background)")

            # Union
            semantic_seg[instance_label] = torch.logical_or(
                semantic_seg[instance_label],
                instance_seg
            )

        # Background: pixels not in any segmentation
        semantic_seg[0] = semantic_seg.sum(dim=0) == 0
        return semantic_seg.float()

    @staticmethod
    def filter_annotations(
        annotations: Sequence['Annotation'],
        type: Literal['label', 'category', 'segmentation', 'all'] = 'all',
        scope: Literal['frame', 'image', 'all'] = 'all',
    ) -> list['Annotation']:
        """Filter annotations by type and scope.

        Args:
            annotations: List of annotations.
            type: Filter by annotation type.
            scope: Filter by scope (frame/image).

        Returns:
            Filtered list of annotations.
        """
        if type not in ['label', 'category', 'segmentation', 'all']:
            raise ValueError(f"Invalid type: {type}")
        if scope not in ['frame', 'image', 'all']:
            raise ValueError(f"Invalid scope: {scope}")

        filtered = []
        for ann in annotations:
            ann_scope = 'image' if ann.frame_index is None else 'frame'
            type_matches = type == 'all' or ann.annotation_type == type
            scope_matches = scope == 'all' or scope == ann_scope

            if type_matches and scope_matches:
                filtered.append(ann)

        return filtered
