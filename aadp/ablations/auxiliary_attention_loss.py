"""AttentionAlignmentLoss and CombinedLoss for the attention supervision ablation.

Implements an auxiliary KL-divergence loss that encourages the A-ADP slice
attention distribution to peak at anatomically grounded slice indices derived
from RadGenome bounding-box annotations.

Usage in training loop::

    criterion = CombinedLoss(lambda_attn=0.1)
    loss = criterion(
        lm_loss,
        attn_weights=projector.get_slice_attention(),  # (B, D)
        gt_slice_indices=gt_indices,                   # list[list[int]]
        use_attn_loss=True,
    )
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionAlignmentLoss(nn.Module):
    """KL divergence between predicted slice attention and a uniform GT target.

    Given ground-truth slice indices (e.g., from RadGenome bounding boxes),
    constructs a uniform distribution over those indices and computes
    KL(target || predicted) summed and normalised over the batch.

    Args:
        lambda_attn: Loss weight. Default 0.1.
        eps:         Epsilon for log stability. Default 1e-8.
    """

    def __init__(self, lambda_attn: float = 0.1, eps: float = 1e-8) -> None:
        super().__init__()
        self.lambda_attn = lambda_attn
        self.eps = eps

    def forward(
        self,
        attn_weights: torch.Tensor,
        gt_slice_indices: List[List[int]],
    ) -> torch.Tensor:
        """Compute the alignment loss.

        Args:
            attn_weights:     ``(B, D)`` — per-slice attention mass from the
                              projector's ``get_slice_attention()`` output.
                              Values must be non-negative and sum to 1 along D
                              (soft-maxed); the method adds ``eps`` before
                              log-softmax for robustness.
            gt_slice_indices: Length-B list of lists.  Each inner list holds
                              the 0-based indices of the ground-truth relevant
                              slices for that batch item.  Empty lists yield
                              zero loss for that item.

        Returns:
            Scalar loss: ``lambda_attn * mean KL over batch``.
        """
        B, D = attn_weights.shape
        device = attn_weights.device

        loss = attn_weights.new_zeros(1)
        n_valid = 0

        for b in range(B):
            indices = gt_slice_indices[b]
            if not indices:
                continue

            # Build uniform target over GT indices
            target = torch.zeros(D, device=device)
            for idx in indices:
                if 0 <= idx < D:
                    target[idx] = 1.0
            if target.sum() == 0:
                continue
            target = target / target.sum()  # normalise

            # Log-softmax predicted distribution (add eps for stability)
            log_pred = F.log_softmax(attn_weights[b] + self.eps, dim=0)

            # KL( target || pred ) = sum( target * (log(target) - log(pred)) )
            # F.kl_div expects log-predictions and raw targets, computes
            # sum( target * (log(target) - input) ) ← reduction='sum'
            kl = F.kl_div(log_pred, target, reduction="sum")
            loss = loss + kl
            n_valid += 1

        if n_valid == 0:
            return attn_weights.new_zeros(1).squeeze()

        return (self.lambda_attn * loss / n_valid).squeeze()


class CombinedLoss(nn.Module):
    """Next-token prediction loss + optional attention alignment auxiliary loss.

    This wrapper makes it trivial to toggle the auxiliary loss on/off for
    ablation studies without modifying the training loop.

    Args:
        lambda_attn: Weight for the alignment term. Default 0.1.
    """

    def __init__(self, lambda_attn: float = 0.1) -> None:
        super().__init__()
        self.attn_loss_fn = AttentionAlignmentLoss(lambda_attn=lambda_attn)

    def forward(
        self,
        lm_loss: torch.Tensor,
        attn_weights: Optional[torch.Tensor] = None,
        gt_slice_indices: Optional[List[List[int]]] = None,
        use_attn_loss: bool = True,
    ) -> torch.Tensor:
        """Combine language model loss with optional attention alignment loss.

        Args:
            lm_loss:          Scalar LM cross-entropy loss from teacher-forcing.
            attn_weights:     ``(B, D)`` slice attention from projector.  If
                              ``None``, attention loss is skipped regardless of
                              ``use_attn_loss``.
            gt_slice_indices: Ground-truth slice index lists (length B).  If
                              ``None``, attention loss is skipped.
            use_attn_loss:    Toggle.  Set ``False`` to run without auxiliary
                              loss even when weights and indices are provided.

        Returns:
            Scalar combined loss.
        """
        if (
            use_attn_loss
            and attn_weights is not None
            and gt_slice_indices is not None
        ):
            aux = self.attn_loss_fn(attn_weights, gt_slice_indices)
            return lm_loss + aux
        return lm_loss
