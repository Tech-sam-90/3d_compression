"""Linear probe for VTCB multi-abnormality classification (T2).

The main model is frozen.  A single linear layer is trained on top of the
mean-pooled M compressed tokens to predict the 18 CT-RATE binary abnormality
labels with a sigmoid output and BCE loss.  Because different token budgets M
produce differently-conditioned representations, a separate probe is trained per
M value; :class:`VTCBRunner` caches trained probes by M so re-running the sweep
does not retrain from scratch.
"""

import logging
from typing import Callable, Dict, Iterable, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LinearProbe(nn.Module):
    """Single linear layer mapping ``(B, C)`` features → ``(B, num_labels)`` logits."""

    def __init__(self, embed_dim: int, num_labels: int = 18) -> None:
        super().__init__()
        self.linear = nn.Linear(embed_dim, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits (no sigmoid — BCEWithLogits / metric applies it)."""
        return self.linear(x)


def train_classification_probe(
    feature_fn: Callable[[torch.Tensor, List[str]], torch.Tensor],
    train_batches: Iterable,
    label_names: List[str],
    instruction: str,
    device: torch.device,
    embed_dim: int,
    max_steps: int = 1000,
    lr: float = 1e-3,
) -> LinearProbe:
    """Train a linear probe on frozen model features.

    Args:
        feature_fn:    ``(volumes, instructions) -> (B, C)`` mean-pooled token
                       features from the frozen model at the current budget M.
        train_batches: Iterable of ``(volumes, reports, label_dicts, pids)``
                       tuples (as produced by ``VTCBRunner._stack_batch`` over a
                       CT-RATE training split).
        label_names:   Ordered CT-RATE label names (length = num_labels).
        instruction:   Instruction string fed to the model during extraction.
        device:        Training device.
        embed_dim:     Feature dimension C.
        max_steps:     Maximum gradient steps. Default 1000.
        lr:            Adam learning rate. Default 1e-3.

    Returns:
        A trained :class:`LinearProbe` in eval mode.
    """
    num_labels = len(label_names)
    probe = LinearProbe(embed_dim, num_labels).to(device)
    optim = torch.optim.Adam(probe.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()

    probe.train()
    step = 0
    while step < max_steps:
        made_progress = False
        for raw_batch in train_batches:
            made_progress = True
            volumes, _, label_dicts, _ = raw_batch
            volumes = volumes.to(device)
            instructions = [instruction] * volumes.shape[0]

            # Features come from the frozen model — no grad through it.
            with torch.no_grad():
                feats = feature_fn(volumes, instructions).detach()

            targets = torch.tensor(
                [[float(ld.get(k, 0.0)) for k in label_names] for ld in label_dicts],
                dtype=torch.float32,
                device=device,
            )

            logits = probe(feats)
            loss = bce(logits, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

            step += 1
            if step >= max_steps:
                break
        if not made_progress:
            # train_batches exhausted and not re-iterable — stop early.
            break

    probe.eval()
    logger.info("Trained classification probe for %d steps.", step)
    return probe
