"""Dice/IoU scoring for IMAGE_SEGMENTATION and VOLUME_SEGMENTATION predictions. """
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from datamint.entities.annotations.base_segmentation import BaseSegmentationAnnotation

_LOGGER = logging.getLogger(__name__)


@dataclass
class SegmentationScores:
    """Dice/IoU results for one evaluate() call, at three levels """

    per_resource: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)
    """resource_id -> class_name -> {'dice': ..., 'iou': ...}. Excludes
    class/resource pairs where ground truth and prediction were both empty
    (nothing to score)."""

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    """class_name -> {'dice': ..., 'iou': ..., 'n': <resources scored>}."""

    dataset: dict[str, float] = field(default_factory=dict)
    """{'dice': ..., 'iou': ...} -- mean of per_class means."""


def _to_bool_mask(annotation: 'BaseSegmentationAnnotation') -> np.ndarray:
    data = annotation.fetch_file_data(auto_convert=True, use_cache=True)
    if hasattr(data, 'get_fdata'):
        
        # nibabel Nifti1Image
        data = data.get_fdata()
    return np.squeeze(np.asarray(data)) > 0


def _combine_masks(anns: list['BaseSegmentationAnnotation'], depth: int | None) -> np.ndarray:
    """Mask for one class on one resource, handling the volume/slice-bridge case. """
    if len(anns) == 1:
        return _to_bool_mask(anns[0])

    frames = {}
    for ann in anns:
        frame_index = ann.frame_index
        if frame_index is None:
            raise ValueError(
                f"Multiple annotations named {ann.name!r} on the same resource, "
                "but not all carry a frame_index -- can't tell which 2D mask "
                "belongs where in the volume."
            )
        frames[frame_index] = _to_bool_mask(ann)

    h, w = next(iter(frames.values())).shape
    depth = depth if depth is not None else max(frames) + 1
    volume = np.zeros((depth, h, w), dtype=bool)
    for frame_index, mask_2d in frames.items():
        volume[frame_index] = mask_2d
    return volume


def _dice_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[float, float] | None:
    """Dice/IoU for one class on one resource, or ``None`` if both masks are empty.

    A model that doesn't predict a class present in ground truth scores 0 for
    that pair
    """
    
    gt_sum = int(gt_mask.sum())
    pred_sum = int(pred_mask.sum())
    if gt_sum == 0 and pred_sum == 0:
        return None

    intersection = int(np.logical_and(gt_mask, pred_mask).sum())
    union = int(np.logical_or(gt_mask, pred_mask).sum())
    dice = (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    return dice, iou


def compute_segmentation_scores(
    resource_ids: Sequence[str],
    ground_truths: Sequence[Sequence['BaseSegmentationAnnotation']],
    predictions: Sequence[Sequence['BaseSegmentationAnnotation']],
) -> SegmentationScores:
    """Score segmentation predictions against ground truth, resource by resource.

    Args:
        resource_ids: Resource ID per index, aligned with ``ground_truths``
            and ``predictions``.
        ground_truths: Ground-truth segmentation annotations per resource.
        predictions: Predicted segmentation annotations per resource, same
            shape as returned by ``BaseDatamintModel.predict()``.
    """
    scores = SegmentationScores()
    per_class_values: dict[str, list[tuple[float, float]]] = {}

    n_resources, n_gts, n_preds = len(resource_ids), len(ground_truths), len(predictions)
    if n_resources != n_gts or n_resources != n_preds:
        _LOGGER.warning(
            "resource_ids (%d), ground_truths (%d), and predictions (%d) have "
            "different lengths; zip() will silently truncate to the shortest.",
            n_resources, n_gts, n_preds,
        )

    for resource_id, gt_anns, pred_anns in zip(resource_ids, ground_truths, predictions):
        gt_by_class: dict[str, list] = {}
        for ann in gt_anns:
            gt_by_class.setdefault(ann.name, []).append(ann)
        pred_by_class: dict[str, list] = {}
        for ann in pred_anns:
            pred_by_class.setdefault(ann.name, []).append(ann)
        class_names = set(gt_by_class) | set(pred_by_class)

        resource_scores: dict[str, dict[str, float]] = {}
        for class_name in class_names:
            gt_anns_for_class = gt_by_class.get(class_name)
            pred_anns_for_class = pred_by_class.get(class_name)
            gt_mask = _combine_masks(gt_anns_for_class, depth=None) if gt_anns_for_class else None
            
            # Reconstruct the prediction at the same depth as ground truth
            pred_depth = gt_mask.shape[0] if gt_mask is not None and gt_mask.ndim == 3 else None
            pred_mask = (
                _combine_masks(pred_anns_for_class, depth=pred_depth) if pred_anns_for_class else None
            )

            if gt_mask is None and pred_mask is None:
                continue
            shape = gt_mask.shape if gt_mask is not None else pred_mask.shape
            if gt_mask is None:
                gt_mask = np.zeros(shape, dtype=bool)
            if pred_mask is None:
                pred_mask = np.zeros(shape, dtype=bool)
            if gt_mask.shape != pred_mask.shape:
                _LOGGER.warning(
                    "Skipping class %r on resource %r: ground truth and prediction "
                    "mask shapes differ (%s vs %s).",
                    class_name, resource_id, gt_mask.shape, pred_mask.shape,
                )
                continue

            result = _dice_iou(gt_mask, pred_mask)
            if result is None:
                continue
            dice, iou = result
            resource_scores[class_name] = {'dice': dice, 'iou': iou}
            per_class_values.setdefault(class_name, []).append((dice, iou))

        if resource_scores:
            scores.per_resource[resource_id] = resource_scores

    for class_name, values in per_class_values.items():
        dices, ious = zip(*values)
        scores.per_class[class_name] = {
            'dice': float(np.mean(dices)),
            'iou': float(np.mean(ious)),
            'n': len(values),
        }

    if scores.per_class:
        scores.dataset = {
            'dice': float(np.mean([c['dice'] for c in scores.per_class.values()])),
            'iou': float(np.mean([c['iou'] for c in scores.per_class.values()])),
        }

    return scores
