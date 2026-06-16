"""Learning rate scheduler with linear warmup + cosine annealing."""

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Linear warmup then cosine annealing LR schedule.

    Phase 1 (warmup): LR rises linearly from 0 → base_lr over
    ``num_warmup_steps`` optimizer steps.

    Phase 2 (cosine): LR falls from base_lr → ``min_lr_ratio * base_lr``
    following a cosine curve over the remaining steps.

    Works with any optimizer regardless of the number of parameter groups
    because ``LambdaLR`` applies the same multiplier to every group's
    base LR.

    Args:
        optimizer:           Target optimizer.
        num_warmup_steps:    Steps for linear warmup.
        num_training_steps:  Total training steps (warmup + cosine).
        min_lr_ratio:        Floor as a fraction of base_lr. Default 0.1.

    Returns:
        ``LambdaLR`` scheduler — call ``.step()`` after each optimizer step.
    """
    def _lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup: 0 → 1
            return float(current_step) / float(max(1, num_warmup_steps))
        # Cosine decay: 1 → min_lr_ratio
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = (1.0 + math.cos(math.pi * progress)) / 2.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, _lr_lambda)
