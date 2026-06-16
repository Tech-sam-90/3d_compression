"""Recall@K metric for slice-level localisation using projector attention.

Measures whether the projector concentrates attention on the ground-truth
slices identified in RadGenome bounding-box annotations.
"""

from typing import Dict, List

import torch


def compute_recall_at_k(
    attn_weights: torch.Tensor,
    gt_slice_indices: List[List[int]],
    k: int,
) -> Dict[str, float]:
    """Compute Recall@K over a batch of volumes.

    For each batch item: rank slices by attention mass (descending), take the
    top-k indices, and check if *any* ground-truth slice is among them.

    Args:
        attn_weights:     ``(B, D)`` per-slice attention mass on any device.
        gt_slice_indices: Length-B list of GT slice index lists (from
                          ``RadGenomeDataset.get_by_patient()``).
                          Empty lists are skipped.
        k:                Number of top slices to examine.  Clamped to D if
                          ``k > D``.

    Returns:
        ``{"recall_at_k": float, "k": int, "n_samples": int}``
    """
    B, D = attn_weights.shape
    k_eff = min(k, D)

    # Top-k slice indices per batch item — on same device as attn_weights
    _, top_k_indices = attn_weights.topk(k_eff, dim=1, largest=True, sorted=False)
    # top_k_indices: (B, k_eff)

    # Move to CPU for Python-level comparison
    top_k_cpu = top_k_indices.cpu().tolist()

    hits = 0
    n_valid = 0
    for b in range(B):
        gt = gt_slice_indices[b]
        if not gt:
            continue  # skip items without GT annotations
        n_valid += 1
        top_set = set(top_k_cpu[b])
        if any(idx in top_set for idx in gt):
            hits += 1

    recall = hits / n_valid if n_valid > 0 else 0.0

    return {
        "recall_at_k": recall,
        "k": k_eff,
        "n_samples": n_valid,
    }


def compute_recall_at_k_curve(
    attn_weights: torch.Tensor,
    gt_slice_indices: List[List[int]],
    k_values: List[int],
) -> Dict[str, float]:
    """Recall@K for multiple K values in a single call.

    Args:
        attn_weights:     ``(B, D)`` per-slice attention mass.
        gt_slice_indices: Ground-truth slice indices (same as
                          :func:`compute_recall_at_k`).
        k_values:         List of K values to evaluate.

    Returns:
        ``{"recall_at_{k}": float}`` for each k in ``k_values``.
    """
    results: Dict[str, float] = {}
    for k in k_values:
        metrics = compute_recall_at_k(attn_weights, gt_slice_indices, k)
        results[f"recall_at_{k}"] = metrics["recall_at_k"]
    return results
