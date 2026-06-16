"""
DetectionDataset - Dataset for object detection tasks.

Loads images and bounding box annotations, returning detection-ready tensors
in pascal_voc (x1, y1, x2, y2) pixel coordinate format.
"""
import logging
from typing import Any
from typing_extensions import override

import numpy as np
import torch

from .image_dataset import ImageDataset

_LOGGER = logging.getLogger(__name__)


class DetectionDataset(ImageDataset):
    """Dataset for 2D object detection.

    Returns items as dicts with keys ``'image'`` (C×H×W tensor),
    ``'boxes'`` (N×4 float32 tensor, pascal_voc pixel coords),
    ``'labels'`` (N int64 tensor), ``'resource_id'`` (str), and
    ``'identifiers'`` (list[str], for v2 instance tracking).

    The class map is built alphabetically from all ``identifier`` values
    found on box annotations across the project, ensuring a stable
    name→index mapping between runs.

    Use :func:`detection_collate_fn` as the DataLoader ``collate_fn``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs['return_segmentations'] = False
        super().__init__(*args, **kwargs)

    @override
    def _setup_dataset(self) -> None:
        super()._setup_dataset()
        self._class_map: dict[str, int] = self._build_class_map()

    def _build_class_map(self) -> dict[str, int]:
        """Return alphabetically-sorted ``{class_name: index}`` from all box annotations."""
        class_names: set[str] = set()
        for anns in self.resource_annotations:
            for ann in anns:
                if getattr(ann, 'annotation_type', None) == 'square':
                    name = ann.identifier
                    if name:
                        class_names.add(name)
        return {name: idx for idx, name in enumerate(sorted(class_names))}

    def _load_image(self, index: int) -> np.ndarray:
        """Load the image at *index* as a (H, W, C) float32 numpy array.

        Delegates to the parent's ``_get_raw_item`` so that format-specific
        handling (DICOM, NIfTI, WebP, uint16 normalisation, …) is not
        duplicated here. 
        """
        
        raw = self._get_raw_item(index)
        img = raw['image']  # (C, N, H, W)
        if img.ndim == 4 and img.shape[1] == 1:
            img = img.squeeze(1)  # (C, H, W)
        elif img.ndim == 4:
            img = img[:, 0]  # first frame (DetectionDataset is 2D-only)
        img = self._preprocess_image_array(img)
        return np.ascontiguousarray(img.transpose(1, 2, 0))  # (H, W, C)

    def _fetch_boxes(
        self,
        resource,
        frame_index: int | None = None,
    ) -> list[tuple[float, float, float, float]]:
        """Return ``(x1, y1, x2, y2)`` pixel-coordinate tuples for box annotations.

        Args:
            resource: Resource with a ``fetch_annotations`` method.
            frame_index: When given, only boxes whose ``frame_index`` matches
                are returned.
        """
        anns = resource.fetch_annotations(annotation_type='square')
        boxes: list[tuple[float, float, float, float]] = []
        for ann in anns:
            if frame_index is not None and ann.frame_index != frame_index:
                continue
            geometry = getattr(ann, 'geometry', None)
            if geometry is None:
                continue
            x1, y1, _ = geometry.point1
            x2, y2, _ = geometry.point2
            boxes.append((float(x1), float(y1), float(x2), float(y2)))
        return boxes

    @override
    def apply_alb_transform(  
        self,
        img: np.ndarray,
        boxes: list[tuple[float, float, float, float]],
        labels: list[int],
        identifiers: list[str] | None = None,
    ) -> dict[str, Any]:
        """Apply albumentations transform with bounding box support.

        The transform must be built with
        ``A.BboxParams(format='pascal_voc', label_fields=['labels', 'identifiers'])``
        so that albumentations keeps identifiers aligned with their boxes when
        spatial transforms remove or reorder boxes.
        """
        if self.alb_transform is None:
            raise ValueError("alb_transform is not set")

        _ids = identifiers if identifiers is not None else ['' for _ in boxes]
        aug = self.alb_transform(image=img, bboxes=boxes, labels=labels, identifiers=_ids)
        aug_img = aug['image']
        aug_boxes: list = list(aug['bboxes'])
        aug_labels: list = list(aug['labels'])
        aug_identifiers: list = list(aug.get('identifiers', _ids[:len(aug_boxes)]))

        if isinstance(aug_img, np.ndarray):
            aug_img = torch.from_numpy(
                np.ascontiguousarray(aug_img.transpose(2, 0, 1)).astype(np.float32)
            )

        if aug_boxes:
            boxes_tensor = torch.tensor(aug_boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(aug_labels, dtype=torch.int64)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            'image': aug_img,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'identifiers': aug_identifiers,
        }

    @override
    def __getitem__(self, index: int) -> dict[str, Any]:
        resource = self.resources[index]
        img_hwc = self._load_image(index)

        # Use pre-fetched annotations 
        anns = self.resource_annotations[index]
        box_anns = [
            ann for ann in anns
            if getattr(ann, 'annotation_type', None) == 'square'
            and getattr(ann, 'geometry', None) is not None
        ]

        # Skip boxes without an identifier 
        unnamed = [ann for ann in box_anns if not ann.identifier]
        if unnamed:
            _LOGGER.warning(
                "Skipping %d box annotation(s) on resource '%s' that have no identifier. "
                "Set a class name on each BoxAnnotation before training.",
                len(unnamed), resource.id,
            )
        box_anns = [ann for ann in box_anns if ann.identifier]

        raw_boxes = [
            (float(ann.geometry.point1[0]), float(ann.geometry.point1[1]),
             float(ann.geometry.point2[0]), float(ann.geometry.point2[1]),
             ann)
            for ann in box_anns
        ]
        degenerate = [(x1, y1, x2, y2) for x1, y1, x2, y2, _ in raw_boxes if x2 <= x1 or y2 <= y1]
        if degenerate:
            _LOGGER.warning(
                "Skipping %d degenerate box(es) on resource '%s' (x2<=x1 or y2<=y1): %s",
                len(degenerate), resource.id, degenerate,
            )
        box_anns = [ann for x1, y1, x2, y2, ann in raw_boxes if x2 > x1 and y2 > y1]
        boxes = [
            (float(ann.geometry.point1[0]), float(ann.geometry.point1[1]),
             float(ann.geometry.point2[0]), float(ann.geometry.point2[1]))
            for ann in box_anns
        ]
        labels = [self._class_map.get(ann.identifier, 0) for ann in box_anns]
        identifiers = [ann.identifier for ann in box_anns]

        if self.alb_transform is not None:
            aug = self.apply_alb_transform(img_hwc, boxes, labels, identifiers)
            img_tensor = aug['image']
            boxes_tensor = aug['boxes']
            labels_tensor = aug['labels']
            identifiers = aug['identifiers']
        else:
            img_tensor = torch.from_numpy(
                np.ascontiguousarray(img_hwc.transpose(2, 0, 1)).astype(np.float32)
            )
            if boxes:
                boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
                labels_tensor = torch.tensor(labels, dtype=torch.int64)
            else:
                boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
                labels_tensor = torch.zeros((0,), dtype=torch.int64)

        return {
            'image': img_tensor,
            'boxes': boxes_tensor,
            'labels': labels_tensor,
            'resource_id': resource.id,
            'identifiers': identifiers,
        }

    @override
    def __repr__(self) -> str:
        base = super(ImageDataset, self).__repr__()
        return f"DetectionDataset\n{base}"


def detection_collate_fn(batch: list[dict]) -> dict:
    """Collate a list of detection items into a batch.

    Images are stacked into a single tensor. Boxes and labels are kept as
    lists because each image may have a different number of annotations.
    """
    return {
        'image': torch.stack([item['image'] for item in batch]),
        'boxes': [item['boxes'] for item in batch],
        'labels': [item['labels'] for item in batch],
        'resource_id': [item['resource_id'] for item in batch],
        'identifiers': [item['identifiers'] for item in batch],
    }
