"""Training loss functions.

``NextTokenLoss`` is the primary LM training signal.
``CombinedLoss`` is re-exported from ``ablations.auxiliary_attention_loss`` so
training code only needs to import from this module.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ablations.auxiliary_attention_loss import CombinedLoss  # re-export

__all__ = ["NextTokenLoss", "CombinedLoss"]


class NextTokenLoss(nn.Module):
    """Cross-entropy loss over the report token positions.

    Inputs must already have ``-100`` on visual and instruction positions;
    only real report token positions contribute to the loss.  This mirrors
    PyTorch's convention for ``nn.CrossEntropyLoss(ignore_index=-100)``.

    Args:
        ignore_index: Token id to ignore. Default ``-100`` (PyTorch default).
    """

    def __init__(self, ignore_index: int = -100) -> None:
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute mean cross-entropy over supervised positions.

        Args:
            logits: ``(B, L, vocab_size)`` — raw LLM logits.
            labels: ``(B, L)`` — target token ids; ``-100`` positions ignored.

        Returns:
            Scalar loss tensor.
        """
        B, L, vocab_size = logits.shape
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=self.ignore_index,
        )
        if torch.isnan(loss):
            raise ValueError(
                "NextTokenLoss is NaN — all labels in this batch may be masked. "
                "Check label construction in MedVLM.forward (visual + instruction "
                "positions should be -100; at least one report token must be real)."
            )
        return loss
