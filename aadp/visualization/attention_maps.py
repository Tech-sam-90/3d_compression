"""Attention map visualisation utilities for A-ADP slice-level analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_attention(
    volume: torch.Tensor,
    attn_weights: torch.Tensor,
    instruction: str,
    save_path: str,
    gt_slice_indices: Optional[List[int]] = None,
) -> None:
    """Visualise per-slice attention alongside top-5 slice thumbnails.

    Two-panel figure:
    - Left:  horizontal bar chart of attention mass per depth slice.
             Ground-truth slices (from RadGenome) are highlighted in teal.
    - Right: montage of the top-5 slices by attention mass with score overlay.

    Args:
        volume:           ``(D, H, W)`` CT volume tensor.
        attn_weights:     ``(D,)`` per-slice attention mass (need not sum to 1).
        instruction:      Clinical question / instruction string.
        save_path:        Filesystem path where the PNG will be written.
        gt_slice_indices: Ground-truth relevant slice indices (RadGenome).
                          Highlighted in teal on the bar chart. ``None`` skips.
    """
    volume = volume.detach().cpu().float()
    attn = attn_weights.detach().cpu().float().numpy()
    D = volume.shape[0]

    gt_set = set(gt_slice_indices or [])
    colours = ["teal" if i in gt_set else "steelblue" for i in range(D)]

    n_thumb = min(5, D)
    top5_idx = np.argsort(attn)[::-1][:n_thumb].tolist()

    fig_height = max(4, D * 0.12 + 2)
    fig = plt.figure(figsize=(14, fig_height))
    gs = gridspec.GridSpec(
        1, 2, figure=fig, wspace=0.35,
        width_ratios=[1.5, n_thumb if n_thumb > 0 else 1],
    )

    # ── Left: horizontal bar chart ─────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0])
    ax_bar.barh(range(D), attn, color=colours, height=0.8)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Attention mass", fontsize=9)
    ax_bar.set_ylabel("Slice index", fontsize=9)
    ax_bar.set_title(
        f'Per-slice attention\n"{instruction[:60]}"',
        fontsize=8, pad=6,
    )
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)
    if gt_set:
        ax_bar.barh([], [], color="teal", label="GT slice")
        ax_bar.barh([], [], color="steelblue", label="Other")
        ax_bar.legend(fontsize=7, loc="lower right")

    # ── Right: top-N thumbnails ────────────────────────────────────────────────
    if n_thumb > 0:
        gs_right = gridspec.GridSpecFromSubplotSpec(
            1, n_thumb, subplot_spec=gs[1], wspace=0.05
        )
        vol_np = volume.numpy()
        vmin, vmax = float(vol_np.min()), float(vol_np.max())
        scale = (vmax - vmin) if vmax != vmin else 1.0

        for col, si in enumerate(top5_idx):
            ax = fig.add_subplot(gs_right[col])
            slice_img = (vol_np[si] - vmin) / scale
            ax.imshow(slice_img, cmap="gray", aspect="equal",
                      interpolation="nearest", vmin=0.0, vmax=1.0)
            ax.set_title(f"#{si}\n{attn[si]:.3f}", fontsize=7)
            ax.axis("off")

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def compare_instructions(
    volume: torch.Tensor,
    instructions: List[str],
    checkpoints: List[Tuple[Any, str]],
    save_path: str,
) -> None:
    """Plot attention distributions for the same volume under different instructions.

    This is the key qualitative figure for the paper: FiLM conditioning steers
    cross-attention to anatomically distinct slices depending on the clinical query.

    Args:
        volume:       ``(D, H, W)`` CT volume tensor.
        instructions: List of instruction strings to compare.
        checkpoints:  ``[(model, model_name), ...]`` — each model is a ``MedVLM``
                      already loaded with the appropriate checkpoint weights.
        save_path:    Filesystem path where the PNG will be written.
    """
    device = next(iter(checkpoints[0][0].parameters())).device
    vol_t = volume.to(device)
    D, H, W = vol_t.shape

    records: List[Tuple[str, str, np.ndarray]] = []

    for model, model_name in checkpoints:
        model.eval()
        slices = vol_t.unsqueeze(1)          # (D, 1, H, W)
        with torch.no_grad():
            patch_tokens = model.vit(slices).float()  # (D, N, C)
        N = patch_tokens.shape[1]
        H_p = W_p = int(math.isqrt(N)) if N > 0 else 1
        patch_tokens = patch_tokens.unsqueeze(0)       # (1, D, N, C)

        for instr in instructions:
            with torch.no_grad():
                etext = model.instruction_encoder([instr]).float()
                model.projector(patch_tokens, etext, H_p, W_p)

            if hasattr(model.projector, "get_slice_attention"):
                raw = model.projector.get_slice_attention()
                if raw is not None:
                    attn_1d = raw[0].cpu().float().numpy()   # (D,)
                else:
                    attn_1d = np.ones(D, dtype=np.float32) / D
            else:
                attn_1d = np.ones(D, dtype=np.float32) / D

            records.append((model_name, instr, attn_1d))

    n_panels = len(records)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(max(4, 5 * n_panels), 5),
        sharey=True,
    )
    if n_panels == 1:
        axes = [axes]

    for i, (ax, (model_name, instr, attn_1d)) in enumerate(zip(axes, records)):
        D_ = len(attn_1d)
        ax.barh(range(D_), attn_1d, color="steelblue", height=0.8)
        ax.invert_yaxis()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Attention mass", fontsize=9)
        ax.set_title(f'{model_name}\n"{instr[:50]}"', fontsize=8, pad=6)
        if i == 0:
            ax.set_ylabel("Slice index", fontsize=9)

    fig.suptitle("Attention distribution per instruction", fontsize=10, y=1.02)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
