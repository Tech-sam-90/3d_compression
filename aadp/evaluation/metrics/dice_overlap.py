"""Dice overlap between projector attention and TotalSegmentator structure masks.

All tensor operations run on the same device as the input tensors (GPU by
default).  Only Python scalars are moved to CPU for the final dict.
"""

import logging
import math
from typing import Dict, List, Optional

import torch

log = logging.getLogger(__name__)

_EPS = 1e-6


def compute_dice_overlap(
    attn_weights: torch.Tensor,
    structure_masks: torch.Tensor,
    threshold: Optional[float] = None,
) -> Dict[str, object]:
    """Dice between binarised per-slice attention and ground-truth structure masks.

    Args:
        attn_weights:    ``(B, D)`` per-slice attention mass (any device).
        structure_masks: ``(B, D, H, W)`` boolean masks from TotalSegmentator.
        threshold:       Binarisation threshold.  ``None`` → per-volume mean
                         attention mass.

    Returns:
        ``{"dice_mean": float, "dice_std": float, "dice_per_sample": List[float]}``
    """
    B, D = attn_weights.shape
    _, D2, H, W = structure_masks.shape
    if D != D2:
        raise ValueError(
            f"attn_weights D ({D}) != structure_masks D ({D2})"
        )

    device = attn_weights.device

    # Broadcast per-slice mass → 3D spatial map (B, D, H, W)
    attn_3d = attn_weights.unsqueeze(-1).unsqueeze(-1).expand(B, D, H, W)

    # Determine threshold per volume: (B, 1, 1, 1)
    if threshold is not None:
        thresh_tensor = torch.full(
            (B, 1, 1, 1), threshold, dtype=attn_3d.dtype, device=device
        )
    else:
        thresh_tensor = attn_weights.mean(dim=1).reshape(B, 1, 1, 1)

    pred_mask = attn_3d > thresh_tensor  # (B, D, H, W) bool

    gt = structure_masks.to(device=device, dtype=torch.bool)

    dice_list: List[float] = []
    for b in range(B):
        p = pred_mask[b]  # (D, H, W)
        g = gt[b]

        g_sum = g.sum().item()
        if g_sum == 0:
            log.debug("structure_masks[%d] is all-zero — Dice set to NaN.", b)
            dice_list.append(float("nan"))
            continue

        intersection = (p & g).sum().float().item()
        p_sum = p.sum().float().item()
        dice = 2.0 * intersection / (p_sum + g_sum + _EPS)
        dice_list.append(dice)

    # Stats over non-NaN items
    valid = [d for d in dice_list if not math.isnan(d)]
    if valid:
        mean_d = sum(valid) / len(valid)
        var_d = sum((d - mean_d) ** 2 for d in valid) / len(valid)
        std_d = math.sqrt(var_d)
    else:
        mean_d = float("nan")
        std_d = float("nan")

    return {
        "dice_mean": mean_d,
        "dice_std": std_d,
        "dice_per_sample": dice_list,
    }


def compute_dice_per_structure(
    attn_weights: torch.Tensor,
    totalseg_dataset,
    patient_ids: List[str],
    structure_names: List[str],
    device: torch.device = torch.device("cuda"),
) -> Dict[str, float]:
    """Dice overlap for each named TotalSegmentator structure.

    Loads masks from ``totalseg_dataset.load_mask()``, runs
    :func:`compute_dice_overlap` for each structure, and returns per-structure
    and macro-averaged Dice scores.

    Args:
        attn_weights:      ``(B, D)`` attention from ``projector.get_slice_attention()``.
        totalseg_dataset:  ``TotalSegmentatorDataset`` instance.
        patient_ids:       Length-B patient ID strings.
        structure_names:   Structure names to evaluate.
        device:            Device for mask tensors.

    Returns:
        ``{"dice_{structure_name}": float, ..., "dice_macro": float}``
    """
    B, D = attn_weights.shape
    structure_dices: Dict[str, float] = {}

    for structure in structure_names:
        masks: List[torch.Tensor] = []
        valid = True

        for pid in patient_ids:
            try:
                mask = totalseg_dataset.load_mask(pid, structure)  # (D', H, W) bool
            except FileNotFoundError:
                log.warning(
                    "Mask for patient '%s' structure '%s' not found — "
                    "skipping structure.",
                    pid, structure,
                )
                valid = False
                break

            # Pad or crop depth dimension to match attn_weights D
            mask_d = mask.shape[0]
            if mask_d < D:
                pad = torch.zeros(
                    D - mask_d, *mask.shape[1:], dtype=torch.bool, device=device
                )
                mask = torch.cat([mask.to(device), pad], dim=0)
            elif mask_d > D:
                mask = mask[:D].to(device)
            else:
                mask = mask.to(device)
            masks.append(mask)

        if not valid:
            structure_dices[f"dice_{structure}"] = float("nan")
            continue

        stacked = torch.stack(masks, dim=0)  # (B, D, H, W)
        result = compute_dice_overlap(attn_weights, stacked)
        structure_dices[f"dice_{structure}"] = result["dice_mean"]

    # Macro average over non-NaN structures
    valid_vals = [v for v in structure_dices.values() if not math.isnan(v)]
    structure_dices["dice_macro"] = (
        sum(valid_vals) / len(valid_vals) if valid_vals else float("nan")
    )

    return structure_dices
