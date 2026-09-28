"""mAP/precision/recall scoring for OBJECT_DETECTION predictions. """
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from datamint.entities.annotations.annotation import Annotation

_LOGGER = logging.getLogger(__name__)

_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALL_POINTS = np.linspace(0.0, 1.0, 101)


@dataclass
class DetectionScores:
    """mAP/precision/recall results for one evaluate() call. """

    per_resource: dict[str, dict[str, Any]] = field(default_factory=dict)
    """resource_id -> {'tp': ..., 'fp': ..., 'fn': ..., 'precision': ..., 'recall': ...} at IoU 0.5.
    Excludes resources with neither ground-truth nor predicted boxes."""

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    """class_name -> {'ap50': ..., 'ap50_95': ..., 'precision': ..., 'recall': ..., 'f1': ...,
    'n': <GT boxes>}. Classes only predicted (n == 0) have just 'precision' and 'n', since
    AP/recall are undefined without ground truth."""

    dataset: dict[str, float] = field(default_factory=dict)
    """{'map50': ..., 'map50_95': ...} -- mean of per_class AP over classes with ground truth."""


def _box(ann: 'Annotation') -> tuple[float, float, float, float] | None:
    """Pixel (x1, y1, x2, y2), read the same way the dataset reads training boxes. """
    geometry = getattr(ann, 'geometry', None)
    if geometry is None:
        return None
    x1, y1, _ = geometry.point1
    x2, y2, _ = geometry.point2
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if x2 <= x1 or y2 <= y1:
        _LOGGER.warning("Skipping degenerate box (x2<=x1 or y2<=y1): (%s, %s, %s, %s)", x1, y1, x2, y2)
        return None
    return x1, y1, x2, y2


def _group_boxes(anns: Sequence['Annotation']) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """class_name -> (boxes (N, 4), confidences (N,)), boxes sorted by descending confidence."""
    grouped: dict[str, list[tuple[tuple[float, float, float, float], float]]] = {}
    for ann in anns:
        box = _box(ann)
        if box is None:
            continue
        grouped.setdefault(ann.identifier, []).append((box, float(getattr(ann, 'confiability', 1.0))))

    result = {}
    for class_name, items in grouped.items():
        boxes = np.array([b for b, _ in items], dtype=float)
        confidences = np.array([c for _, c in items], dtype=float)
        order = np.argsort(-confidences, kind='stable')
        result[class_name] = (boxes[order], confidences[order])
    return result


def _iou_matrix(pred_boxes: np.ndarray, gt_boxes: np.ndarray) -> np.ndarray:
    """(P, G) IoU between every predicted and ground-truth box."""
    x1 = np.maximum(pred_boxes[:, None, 0], gt_boxes[None, :, 0])
    y1 = np.maximum(pred_boxes[:, None, 1], gt_boxes[None, :, 1])
    x2 = np.minimum(pred_boxes[:, None, 2], gt_boxes[None, :, 2])
    y2 = np.minimum(pred_boxes[:, None, 3], gt_boxes[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    pred_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
    gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
    union = pred_area[:, None] + gt_area[None, :] - intersection
    return np.where(union > 0, intersection / union, 0.0)


def _match(ious: np.ndarray, threshold: float) -> np.ndarray:
    """TP flag per prediction (already sorted by confidence) """
    is_tp = np.zeros(ious.shape[0], dtype=bool)
    if ious.shape[1] == 0:
        return is_tp
    matched = np.zeros(ious.shape[1], dtype=bool)
    for p in range(ious.shape[0]):
        candidates = np.where(matched, -1.0, ious[p])
        g = int(np.argmax(candidates))
        if candidates[g] >= threshold:
            is_tp[p] = True
            matched[g] = True
    return is_tp


def _average_precision(confidences: np.ndarray, is_tp: np.ndarray, n_gt: int) -> float:
    """101-point interpolated AP. """
    if n_gt == 0 or len(confidences) == 0:
        return 0.0
    order = np.argsort(-confidences, kind='stable')
    tp_cum_sum = np.cumsum(is_tp[order])
    fp_cum_sum = np.cumsum(~is_tp[order])
    recall = tp_cum_sum / n_gt
    precision = tp_cum_sum / (tp_cum_sum + fp_cum_sum)
    
    #best precision achievable at this recall or higher
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, _RECALL_POINTS, side='left')
    
    sampled = np.where(idx < len(precision), precision[np.minimum(idx, len(precision) - 1)], 0.0)
    return float(np.mean(sampled))


def compute_detection_scores(
    resource_ids: Sequence[str],
    ground_truths: Sequence[Sequence['Annotation']],
    predictions: Sequence[Sequence['Annotation']],
) -> DetectionScores:
    """Score box predictions against ground truth.

    Boxes are matched per class (``identifier``), one-to-one, highest confidence
    first. A class with ground truth but no predictions scores AP 0; a class only
    predicted is reported but excluded from mAP.

    Args:
        resource_ids: Resource ID per index, aligned with ``ground_truths``
            and ``predictions``.
        ground_truths: Ground-truth box annotations per resource.
        predictions: Predicted box annotations per resource, same shape as
            returned by ``BaseDatamintModel.predict()``. ``confiability`` is
            used as the ranking score.
    """
    scores = DetectionScores()

    n_resources, n_gts, n_preds = len(resource_ids), len(ground_truths), len(predictions)
    if n_resources != n_gts or n_resources != n_preds:
        _LOGGER.warning(
            "resource_ids (%d), ground_truths (%d), and predictions (%d) have "
            "different lengths; zip() will silently truncate to the shortest.",
            n_resources, n_gts, n_preds,
        )

    patient_boxes = sum(
        getattr(getattr(a, 'geometry', None), 'coordinate_system', None) == 'patient'
        for anns in (*ground_truths, *predictions) for a in anns
    )
    if patient_boxes:
        _LOGGER.warning(
            "%d box(es) are in patient coordinates; they are read as pixel coordinates "
            "(same as the training dataset), so IoU may be wrong for them.", patient_boxes,
        )

    # class_name -> per-threshold lists of (confidences, is_tp) arrays, plus GT count
    class_confidences: dict[str, list[np.ndarray]] = {}
    class_is_tp: dict[str, list[np.ndarray]] = {}
    class_n_gt: dict[str, int] = {}

    for resource_id, gt_anns, pred_anns in zip(resource_ids, ground_truths, predictions):
        gt_by_class = _group_boxes(gt_anns)
        pred_by_class = _group_boxes(pred_anns)
        resource_counts = {'tp': 0, 'fp': 0, 'fn': 0}

        for class_name in set(gt_by_class) | set(pred_by_class):
            gt_boxes = gt_by_class.get(class_name, (np.zeros((0, 4)), np.zeros(0)))[0]
            pred_boxes, confidences = pred_by_class.get(class_name, (np.zeros((0, 4)), np.zeros(0)))
            ious = _iou_matrix(pred_boxes, gt_boxes)
            # (n_thresholds, n_preds) TP flags, one row per IoU threshold
            is_tp = np.zeros((len(_IOU_THRESHOLDS), len(pred_boxes)), dtype=bool)
            for i, threshold in enumerate(_IOU_THRESHOLDS):
                is_tp[i] = _match(ious, threshold)

            class_confidences.setdefault(class_name, []).append(confidences)
            class_is_tp.setdefault(class_name, []).append(is_tp)
            class_n_gt[class_name] = class_n_gt.get(class_name, 0) + len(gt_boxes)

            tp = int(is_tp[0].sum())
            resource_counts['tp'] += tp
            resource_counts['fp'] += len(pred_boxes) - tp
            resource_counts['fn'] += len(gt_boxes) - tp

        tp, fp, fn = resource_counts['tp'], resource_counts['fp'], resource_counts['fn']
        if tp + fp + fn == 0:
            continue
        scores.per_resource[resource_id] = {
            **resource_counts,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0.0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        }

    for class_name in sorted(class_n_gt):
        confidences = np.concatenate(class_confidences[class_name])
        is_tp = np.concatenate(class_is_tp[class_name], axis=1)
        n_gt = class_n_gt[class_name]

        tp = int(is_tp[0].sum())
        fp = len(confidences) - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if n_gt == 0:
            scores.per_class[class_name] = {'precision': precision, 'n': 0}
            continue

        recall = tp / n_gt
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        aps = [_average_precision(confidences, is_tp[i], n_gt) for i in range(len(_IOU_THRESHOLDS))]
        scores.per_class[class_name] = {
            'ap50': aps[0],
            'ap50_95': float(np.mean(aps)),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n': n_gt,
        }

    scored = [c for c in scores.per_class.values() if c['n'] > 0]
    if scored:
        scores.dataset = {
            'map50': float(np.mean([c['ap50'] for c in scored])),
            'map50_95': float(np.mean([c['ap50_95'] for c in scored])),
        }

    return scores
