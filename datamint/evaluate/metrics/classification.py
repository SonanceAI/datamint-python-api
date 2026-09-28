"""Accuracy/F1 scoring for single-label IMAGE_CLASSIFICATION predictions. """
from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from datamint.entities.annotations.annotation import Annotation

_LOGGER = logging.getLogger(__name__)


@dataclass
class ClassificationScores:
    """Accuracy/F1 results for one evaluate() call. """

    per_resource: dict[str, dict[str, Any]] = field(default_factory=dict)
    """resource_id -> {'true': ..., 'predicted': ..., 'correct': ...}."""

    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    """class_name -> {'precision': ..., 'recall': ..., 'f1': ..., 'n': <GT instances>}."""

    dataset: dict[str, float] = field(default_factory=dict)
    """{'accuracy': ..., 'f1': ...} -- f1 is the macro average of per_class f1."""


def _class_key(ann: 'Annotation') -> str:
    """Class identity for a category annotation. """
    return ann.value if ann.value is not None else ann.name


def compute_classification_scores(
    resource_ids: Sequence[str],
    ground_truths: Sequence[Sequence['Annotation']],
    predictions: Sequence[Sequence['Annotation']],
) -> ClassificationScores:
    """Score single-label classification predictions against ground truth.

    Args:
        resource_ids: Resource ID per index, aligned with ``ground_truths``
            and ``predictions``.
        ground_truths: Ground-truth category annotations per resource.
            Exactly one is expected per resource -- 0 or >1 are skipped.
        predictions: Predicted category annotations per resource, same
            shape as returned by ``BaseDatamintModel.predict()``. 0 (e.g.
            filtered out by ``confidence_threshold``) counts as incorrect;
            >1 uses the first (logged) since the model contract is one
            prediction per resource.
    """
    scores = ClassificationScores()

    n_resources, n_gts, n_preds = len(resource_ids), len(ground_truths), len(predictions)
    if n_resources != n_gts or n_resources != n_preds:
        _LOGGER.warning(
            "resource_ids (%d), ground_truths (%d), and predictions (%d) have "
            "different lengths; zip() will silently truncate to the shortest.",
            n_resources, n_gts, n_preds,
        )

    class_counts: dict[str, dict[str, int]] = {}
    n_correct = 0
    n_scored = 0

    for resource_id, gt_anns, pred_anns in zip(resource_ids, ground_truths, predictions):
        if len(gt_anns) == 0:
            continue
        if len(gt_anns) > 1:
            _LOGGER.warning(
                "Resource %r has %d ground-truth classification labels; single-label "
                "scoring needs exactly one. Skipping.", resource_id, len(gt_anns),
            )
            continue
        gt_class = _class_key(gt_anns[0])

        if len(pred_anns) > 1:
            _LOGGER.warning(
                "Resource %r has %d predicted classification labels; expected at "
                "most one. Using the first.", resource_id, len(pred_anns),
            )
        pred_class = _class_key(pred_anns[0]) if pred_anns else None

        n_scored += 1
        is_correct = pred_class == gt_class
        if is_correct:
            n_correct += 1

        scores.per_resource[resource_id] = {
            'true': gt_class,
            'predicted': pred_class,
            'correct': is_correct,
        }

        class_counts.setdefault(gt_class, {'tp': 0, 'fp': 0, 'fn': 0})
        if is_correct:
            class_counts[gt_class]['tp'] += 1
        else:
            class_counts[gt_class]['fn'] += 1
            if pred_class is not None:
                class_counts.setdefault(pred_class, {'tp': 0, 'fp': 0, 'fn': 0})
                class_counts[pred_class]['fp'] += 1

    for class_name, counts in class_counts.items():
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        scores.per_class[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'n': tp + fn,
        }

    if n_scored > 0:
        scores.dataset = {
            'accuracy': n_correct / n_scored,
            'f1': float(np.mean([c['f1'] for c in scores.per_class.values()])) if scores.per_class else 0.0,
        }

    return scores
