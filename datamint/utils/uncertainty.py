"""Predictive-entropy uncertainty utilities.

Computes how "spread out" a model's own softmax/sigmoid output is. """
from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-8


def categorical_entropy(probs: Tensor, dim: int = -1) -> Tensor:
    """Normalized Shannon entropy of a categorical probability distribution.

    Args:
        probs: Probabilities that sum to 1 along ``dim`` (e.g. softmax output).
        dim: Dimension the distribution lies along.

    Returns:
        Entropy in ``[0, 1]``, one value per remaining dimension. ``0`` means
        the distribution is fully concentrated on one class (certain); ``1``
        means it is uniform across all classes (maximally uncertain).
    """
    num_classes = probs.shape[dim]
    ent = -(probs * torch.log(probs + _EPS)).sum(dim=dim)
    return ent / torch.log(torch.tensor(float(num_classes), device=probs.device))


def binary_entropy(probs: Tensor) -> Tensor:
    """Normalized binary entropy, element-wise, for sigmoid probabilities.

    Args:
        probs: Sigmoid probabilities in ``[0, 1]``, any shape.

    Returns:
        Entropy in ``[0, 1]`` with the same shape as ``probs``. ``0`` at
        ``p=0`` or ``p=1`` (certain), ``1`` at ``p=0.5`` (maximally uncertain).
    """
    ent = -(probs * torch.log(probs + _EPS) + (1 - probs) * torch.log(1 - probs + _EPS))
    return ent / torch.log(torch.tensor(2.0, device=probs.device))


def segmentation_uncertainty(probs: Tensor, pred_mask: Tensor) -> float | None:
    """Uncertainty score for one class channel of a segmentation prediction.

    Averages per-pixel binary entropy over the pixels the model predicted as
    foreground for this class. 
    
    Args:
        probs: Sigmoid probabilities for this class, any shape (e.g. ``(H,
            W)`` for a 2-D slice).
        pred_mask: Boolean/binary mask of the same shape marking pixels the
            model predicted as this class.

    Returns:
        Mean entropy over the predicted-foreground pixels, or ``None`` if the
        model predicted no foreground pixels at all for this class.
    """
    fg = pred_mask.bool()
    if not fg.any():
        return None
    ent = binary_entropy(probs)
    return float(ent[fg].mean())


def pool_top_k(scores: list[float | None], top_fraction: float = 0.2) -> float | None:
    """Collapse many small scores into one representative score.

    Averages only the top ``top_fraction`` highest scores (e.g. the most
    uncertain slices in a volume).

    Args:
        scores: Per-item scores; ``None`` entries (e.g. an item with nothing
            predicted) are skipped.
        top_fraction: Fraction of the (non-``None``) scores to average, in
            ``(0, 1]``.

    Returns:
        The mean of the top ``top_fraction`` scores, or ``None`` if every
        entry was ``None``.
    """
    if not 0 < top_fraction <= 1:
        raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}")

    valid = [s for s in scores if s is not None]
    if not valid:
        return None

    arr = torch.tensor(valid, dtype=torch.float32)
    k = max(1, int(len(arr) * top_fraction))
    top_k = torch.topk(arr, k).values
    return float(top_k.mean())
