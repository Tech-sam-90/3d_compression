"""AUROC and F1 metrics for CT-RATE multi-abnormality classification.

Uses scikit-learn for AUROC / F1 computation; inputs can be GPU tensors
(moved to CPU automatically).
"""

import logging
import warnings
from typing import Dict, List

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def compute_auroc_f1(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    label_names: List[str],
) -> Dict[str, float]:
    """Per-label AUROC and F1 for multi-label abnormality classification.

    Args:
        predictions: ``(N, num_labels)`` sigmoid probabilities on any device.
        labels:      ``(N, num_labels)`` ground-truth binary labels (0 or 1).
        label_names: List of length ``num_labels`` with human-readable names.

    Returns:
        Dict with:
            ``"auroc_{name}"`` for each label,
            ``"f1_{name}"`` for each label,
            ``"auroc_macro"`` — macro AUROC (NaN labels excluded),
            ``"f1_macro"`` — macro F1 at threshold 0.5.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    # Move to CPU for sklearn
    preds_np = predictions.detach().cpu().float().numpy()
    labels_np = labels.detach().cpu().float().numpy()

    N, num_labels = preds_np.shape
    if len(label_names) != num_labels:
        raise ValueError(
            f"label_names length ({len(label_names)}) must match "
            f"predictions.shape[1] ({num_labels})"
        )

    results: Dict[str, float] = {}
    auroc_values: List[float] = []
    f1_values: List[float] = []

    for i, name in enumerate(label_names):
        col_labels = labels_np[:, i]
        col_preds = preds_np[:, i]

        # AUROC requires at least one positive example
        if col_labels.sum() == 0:
            log.warning(
                "Label '%s' has no positive examples in this batch — "
                "AUROC set to NaN.",
                name,
            )
            auroc = float("nan")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                auroc = float(roc_auc_score(col_labels, col_preds))
        results[f"auroc_{name}"] = auroc

        # F1 at threshold 0.5
        col_bin = (col_preds >= 0.5).astype(int)
        f1 = float(f1_score(col_labels, col_bin, zero_division=0))
        results[f"f1_{name}"] = f1

        if auroc == auroc:  # not NaN
            auroc_values.append(auroc)
        f1_values.append(f1)

    results["auroc_macro"] = float(
        sum(auroc_values) / len(auroc_values) if auroc_values else float("nan")
    )
    results["f1_macro"] = float(
        sum(f1_values) / len(f1_values) if f1_values else float("nan")
    )

    return results


class AbnormalityClassificationHead(nn.Module):
    """Linear probe: mean-pooled visual tokens → per-label sigmoid probabilities.

    Used for the Multi-Abnormality Classification task in the VTCB benchmark.

    Args:
        embed_dim:  Projector output dimension C.
        num_labels: Number of binary abnormality labels. Default 18 (CT-RATE).
    """

    def __init__(self, embed_dim: int, num_labels: int = 18) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map mean-pooled tokens to label probabilities.

        Args:
            x: ``(B, C)`` — mean-pooled visual tokens from the projector.

        Returns:
            ``(B, num_labels)`` probabilities in ``[0, 1]``.
        """
        return torch.sigmoid(self.linear(x))
