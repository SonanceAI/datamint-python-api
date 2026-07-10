"""Inter-annotator agreement metrics for Datamint annotations.

Computes how well multiple annotators agree on the same resources, using a
metric appropriate to the annotation type: Dice for segmentations, IoU for
boxes, Cohen's/Fleiss' kappa for category/label annotations.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Literal, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from datamint.entities.annotations import Annotation, AnnotationType, BoxAnnotation
from datamint.entities.annotations.base_segmentation import BaseSegmentationAnnotation
from datamint.entities.annotations.geometry import BoxGeometry

MetricName = Literal['auto', 'dice', 'iou', 'cohen_kappa', 'fleiss_kappa']

_PER_PAIR_GEOMETRIC_COLUMNS = [
    'resource_id', 'identifier', 'frame_index', 'annotator_a', 'annotator_b', 'metric', 'score',
]
_PER_PAIR_CATEGORICAL_COLUMNS = [
    'resource_id', 'identifier', 'frame_index', 'annotator_a', 'annotator_b',
    'label_a', 'label_b', 'metric', 'score', 'used_in_overall',
]


@dataclass
class AgreementResult:
    """Result of :func:`compute_agreement`.

    Attributes:
        per_pair: One row per compared annotator pair per item. For
            cohen_kappa/fleiss_kappa, includes a ``used_in_overall`` column.
            For fleiss_kappa, an item is used only if its rater count matches
            the most common rater count across all items (Fleiss' kappa
            requires a consistent count, not the same rater identities every
            time), so a low-scoring row with ``used_in_overall=False`` may
            not be reflected in ``overall``.
        per_resource_mean: Mean score per ``(resource_id, identifier)``.
        overall: Summary agreement score for the whole input (mean of
            per-resource means for Dice/IoU; a single kappa value computed
            over all items for cohen_kappa/fleiss_kappa).
        flagged: Rows of ``per_resource_mean`` below the requested threshold.
            Empty (but present) when no threshold was given.
    """

    per_pair: pd.DataFrame
    per_resource_mean: pd.DataFrame
    overall: float
    flagged: pd.DataFrame


def dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks of identical shape."""
    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes differ: {mask_a.shape} vs {mask_b.shape}. "
            "Masks must be spatially aligned to compute Dice."
        )
    mask_a = mask_a.astype(bool)
    mask_b = mask_b.astype(bool)
    denom = int(mask_a.sum()) + int(mask_b.sum())
    if denom == 0:
        return 1.0
    intersection = int(np.logical_and(mask_a, mask_b).sum())
    return 2 * intersection / denom


def iou_boxes(box_a: BoxGeometry, box_b: BoxGeometry) -> float:
    """IoU between the axis-aligned bounding rectangles of two box geometries.

    Both boxes must use the same ``coordinate_system``.
    """
    if box_a.coordinate_system != box_b.coordinate_system:
        raise ValueError(
            "Cannot compare boxes in different coordinate systems: "
            f"{box_a.coordinate_system!r} vs {box_b.coordinate_system!r}."
        )
    ax1, ay1, ax2, ay2 = _box_extent(box_a)
    bx1, by1, bx2, by2 = _box_extent(box_b)

    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - intersection
    if union == 0:
        return 1.0
    return intersection / union


def _box_extent(geom: BoxGeometry) -> tuple[float, float, float, float]:
    xs = [p[0] for p in geom.points]
    ys = [p[1] for p in geom.points]
    return min(xs), min(ys), max(xs), max(ys)


def cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa between two annotators' labels over the same items."""
    if len(labels_a) != len(labels_b):
        raise ValueError("labels_a and labels_b must have the same length.")
    n = len(labels_a)
    if n == 0:
        raise ValueError("Cannot compute Cohen's kappa on zero items.")

    categories = sorted(set(labels_a) | set(labels_b))
    index = {category: i for i, category in enumerate(categories)}
    confusion = np.zeros((len(categories), len(categories)))
    for label_a, label_b in zip(labels_a, labels_b):
        confusion[index[label_a], index[label_b]] += 1

    po = np.trace(confusion) / n
    row_marginal = confusion.sum(axis=1) / n
    col_marginal = confusion.sum(axis=0) / n
    pe = float(np.dot(row_marginal, col_marginal))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings: Sequence[Sequence[str]]) -> float:
    """Fleiss' kappa across 3+ annotators.

    Args:
        ratings: One entry per item, each a sequence of category labels (one
            per annotator). Every item must have the same number of
            annotators.
    """
    n_items = len(ratings)
    if n_items == 0:
        raise ValueError("Cannot compute Fleiss' kappa on zero items.")
    n_raters = len(ratings[0])
    if n_raters < 2 or any(len(item) != n_raters for item in ratings):
        raise ValueError("Fleiss' kappa requires the same number (2+) of raters for every item.")

    categories = sorted({label for item in ratings for label in item})
    index = {category: i for i, category in enumerate(categories)}
    counts = np.zeros((n_items, len(categories)))
    for i, item in enumerate(ratings):
        for label in item:
            counts[i, index[label]] += 1

    p_i = (np.sum(counts * counts, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = float(p_i.mean())
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    pe_bar = float(np.sum(p_j * p_j))
    if pe_bar == 1.0:
        return 1.0
    return (p_bar - pe_bar) / (1 - pe_bar)


def _kind(annotation: Annotation) -> str:
    if isinstance(annotation, BaseSegmentationAnnotation) or annotation.annotation_type == AnnotationType.SEGMENTATION.value:
        return 'segmentation'
    if isinstance(annotation, BoxAnnotation) or annotation.annotation_type == AnnotationType.SQUARE.value:
        return 'square'
    if annotation.annotation_type in (AnnotationType.CATEGORY.value, AnnotationType.LABEL.value):
        return 'category'
    return 'other'


def _group_by_item(annotations: Sequence[Annotation]) -> dict[tuple, dict[str, Annotation]]:
    """Group annotations by (resource_id, identifier, frame_index), then by annotator."""
    groups: dict[tuple, dict[str, Annotation]] = {}
    for annotation in annotations:
        if annotation.created_by is None:
            continue
        key = (annotation.resource_id, annotation.identifier, annotation.frame_index)
        by_annotator = groups.setdefault(key, {})
        if annotation.created_by in by_annotator:
            continue  # multiple annotations from the same annotator for the same item; keep the first
        by_annotator[annotation.created_by] = annotation
    return groups


def _to_binary_mask(data) -> np.ndarray:
    if isinstance(data, Image.Image):
        arr = np.array(data)
    elif isinstance(data, np.ndarray):
        arr = data
    else:  # nibabel Nifti1Image
        arr = np.asarray(data.dataobj)
    return arr > 0


def _flag(per_resource_mean: pd.DataFrame, threshold: float | None) -> pd.DataFrame:
    if threshold is None:
        return per_resource_mean.iloc[0:0]
    return per_resource_mean[per_resource_mean['score'] < threshold]


def _compute_geometric_agreement(
    annotations: Sequence[Annotation],
    metric: Literal['dice', 'iou'],
    threshold: float | None,
) -> AgreementResult:
    groups = _group_by_item(annotations)
    rows = []
    for (resource_id, identifier, frame_index), by_annotator in groups.items():
        if len(by_annotator) < 2:
            continue
        for annotator_a, annotator_b in combinations(sorted(by_annotator), 2):
            ann_a, ann_b = by_annotator[annotator_a], by_annotator[annotator_b]
            if metric == 'dice':
                mask_a = _to_binary_mask(ann_a.fetch_file_data(use_cache=True))
                mask_b = _to_binary_mask(ann_b.fetch_file_data(use_cache=True))
                score = dice_coefficient(mask_a, mask_b)
            else:
                score = iou_boxes(ann_a.geometry, ann_b.geometry)
            rows.append({
                'resource_id': resource_id, 'identifier': identifier, 'frame_index': frame_index,
                'annotator_a': annotator_a, 'annotator_b': annotator_b,
                'metric': metric, 'score': score,
            })

    if not rows:
        raise ValueError("No resource/label groups have 2+ distinct annotators; nothing to compare.")

    per_pair = pd.DataFrame(rows, columns=_PER_PAIR_GEOMETRIC_COLUMNS)
    per_resource_mean = per_pair.groupby(['resource_id', 'identifier'], as_index=False)['score'].mean()
    overall = float(per_resource_mean['score'].mean())
    return AgreementResult(
        per_pair=per_pair,
        per_resource_mean=per_resource_mean,
        overall=overall,
        flagged=_flag(per_resource_mean, threshold),
    )


def _compute_categorical_agreement(
    annotations: Sequence[Annotation],
    metric: Literal['auto', 'cohen_kappa', 'fleiss_kappa'],
    threshold: float | None,
) -> AgreementResult:
    groups = _group_by_item(annotations)
    comparable_groups = {key: by_annotator for key, by_annotator in groups.items() if len(by_annotator) >= 2}
    if not comparable_groups:
        raise ValueError("No resource/label groups have 2+ distinct annotators; nothing to compare.")

    all_annotators = sorted({a for by_annotator in comparable_groups.values() for a in by_annotator})
    if metric == 'auto':
        metric = 'cohen_kappa' if len(all_annotators) == 2 else 'fleiss_kappa'

    if metric == 'cohen_kappa' and len(all_annotators) != 2:
        raise ValueError(
            f"cohen_kappa requires exactly 2 annotators, found {len(all_annotators)}: {all_annotators}. "
            "Use metric='fleiss_kappa' for 3+ annotators."
        )

    # Fleiss' kappa only requires a consistent count of raters per item, not the same
    # rater identities every time
    n_star = None
    if metric == 'fleiss_kappa':
        rater_counts = [len(by_annotator) for by_annotator in comparable_groups.values()]
        
        # Find the most common rater count (n_star) across all items
        n_star = max(set(rater_counts), key=lambda n: (rater_counts.count(n), n))

    rows = []
    for (resource_id, identifier, frame_index), by_annotator in comparable_groups.items():
        used_in_overall = True if n_star is None else len(by_annotator) == n_star
        for annotator_a, annotator_b in combinations(sorted(by_annotator), 2):
            label_a = by_annotator[annotator_a].text_value
            label_b = by_annotator[annotator_b].text_value
            rows.append({
                'resource_id': resource_id, 'identifier': identifier, 'frame_index': frame_index,
                'annotator_a': annotator_a, 'annotator_b': annotator_b,
                'label_a': label_a, 'label_b': label_b,
                'score': float(label_a == label_b),
                'used_in_overall': used_in_overall,
            })

    if metric == 'cohen_kappa':
        labels_a = [row['label_a'] for row in rows]
        labels_b = [row['label_b'] for row in rows]
        overall = cohen_kappa(labels_a, labels_b)
    else:
        ratings = [
            [ann.text_value for ann in by_annotator.values()]
            for by_annotator in comparable_groups.values()
            if len(by_annotator) == n_star
        ]
        overall = fleiss_kappa(ratings)

    for row in rows:
        row['metric'] = metric
    per_pair = pd.DataFrame(rows, columns=_PER_PAIR_CATEGORICAL_COLUMNS)
    per_resource_mean = per_pair.groupby(['resource_id', 'identifier'], as_index=False).agg(
        score=('score', 'mean'),
        used_in_overall=('used_in_overall', 'all'),
    )
    return AgreementResult(
        per_pair=per_pair,
        per_resource_mean=per_resource_mean,
        overall=overall,
        flagged=_flag(per_resource_mean, threshold),
    )


def compute_agreement(
    annotations: Sequence[Annotation],
    metric: MetricName = 'auto',
    threshold: float | None = None,
) -> AgreementResult:
    """Compute inter-annotator agreement over a set of annotations.

    Groups annotations by ``(resource_id, identifier, frame_index)``, then
    compares every pair of annotators on each group with a metric appropriate
    to the annotation type.

    Args:
        annotations: Annotations to compare, typically fetched with
            ``api.annotations.get_list(worklist_id=..., annotation_type=...)``.
            All annotations must share the same "kind" (all segmentations, all
            boxes, or all category/label) unless ``metric`` is given
            explicitly, since a fair comparison metric can't be picked
            automatically across mixed types.
        metric: ``'auto'`` picks Dice for segmentations, IoU for boxes, and
            Cohen's kappa (2 annotators) or Fleiss' kappa (3+) for
            category/label annotations. Pass one explicitly to override.
        threshold: Optional score cutoff. Rows of ``per_resource_mean`` below
            it are returned in ``AgreementResult.flagged`` for adjudication.

    Returns:
        AgreementResult with per-pair scores, per-resource means, an overall
        summary score, and any flagged low-agreement resources.
    """
    if not annotations:
        raise ValueError("No annotations provided.")

    kinds = {_kind(a) for a in annotations}

    if metric in ('dice', 'iou'):
        return _compute_geometric_agreement(annotations, metric, threshold)
    if metric in ('cohen_kappa', 'fleiss_kappa'):
        return _compute_categorical_agreement(annotations, metric, threshold)

    # metric == 'auto': infer the family from the annotation kinds present.
    if kinds == {'segmentation'}:
        return _compute_geometric_agreement(annotations, 'dice', threshold)
    if kinds == {'square'}:
        return _compute_geometric_agreement(annotations, 'iou', threshold)
    if kinds <= {'category'}:
        return _compute_categorical_agreement(annotations, 'auto', threshold)

    raise ValueError(
        f"Cannot auto-select a metric for mixed/unsupported annotation kinds {sorted(kinds)}. "
        "Filter to a single annotation_type first, or pass metric explicitly."
    )
