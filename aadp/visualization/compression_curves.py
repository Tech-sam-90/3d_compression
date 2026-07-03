"""Publication-quality compression–quality curve plotter.

Loads ``*_vtcb.json`` result files and generates one figure per metric plus a
2×2 summary grid, saved as both PNG (presentations) and PDF (LaTeX).
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


_METRIC_LABELS: Dict[str, str] = {
    "radgraph_f1": "RadGraph F1",
    "radgraph_precision": "RadGraph Precision",
    "radgraph_recall": "RadGraph Recall",
    "ratescore_mean": "RaTEScore",
    "ratescore_std": "RaTEScore std",
    "auroc_macro": "AUROC (macro)",
    "f1_macro": "F1 (macro)",
    "recall_at_5": "Recall@5",
    "recall_at_10": "Recall@10",
    "dice_macro": "Dice (macro)",
}

_PRIMARY_METRICS = ["radgraph_f1", "ratescore_mean", "auroc_macro", "recall_at_5"]

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]


def _flatten(task_results: Dict[str, Any]) -> Dict[str, float]:
    flat: Dict[str, float] = {}
    for k, v in task_results.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            for mk, mv in v.items():
                if not mk.startswith("_") and mv is not None:
                    try:
                        flat[mk] = float(mv)
                    except (TypeError, ValueError):
                        pass
        elif isinstance(v, (int, float)) and v is not None:
            try:
                flat[k] = float(v)
            except (TypeError, ValueError):
                pass
    return flat


def _load_results(results_dir: str) -> Dict[str, Any]:
    loaded: Dict[str, Any] = {}
    for p in sorted(Path(results_dir).glob("*_vtcb.json")):
        with open(p) as fh:
            data = json.load(fh)
        name = data.get("model_name", p.stem.replace("_vtcb", ""))
        loaded[name] = data
    return loaded


def _collect_metrics(loaded: Dict[str, Any]) -> List[str]:
    seen: List[str] = []
    for data in loaded.values():
        for M_str, task_res in data.get("results", {}).items():
            if M_str.startswith("_"):
                continue
            for m in _flatten(task_res):
                if m not in seen:
                    seen.append(m)
    return seen


def _apply_paper_style(
    ax: plt.Axes, xlabel: str, ylabel: str, title: str
) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, pad=8)
    ax.tick_params(labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)


def _plot_metric_on_ax(
    ax: plt.Axes,
    loaded: Dict[str, Any],
    metric_name: str,
) -> None:
    for i, (model_name, data) in enumerate(loaded.items()):
        results = data.get("results", {})
        budgets = sorted(int(k) for k in results if not k.startswith("_"))
        values = [
            _flatten(results.get(str(M), {})).get(metric_name, float("nan"))
            for M in budgets
        ]
        valid_pairs = [
            (m, v) for m, v in zip(budgets, values)
            if not (isinstance(v, float) and math.isnan(v))
        ]
        if not valid_pairs:
            continue
        xs, ys = zip(*valid_pairs)
        ax.plot(
            xs, ys,
            marker=_MARKERS[i % len(_MARKERS)],
            color=_COLORS[i % len(_COLORS)],
            label=model_name,
            linewidth=1.5,
            markersize=6,
        )


def _save(fig: plt.Figure, save_dir: str, stem: str) -> None:
    safe = stem.replace("/", "_")
    fig.savefig(
        os.path.join(save_dir, f"{safe}.png"),
        dpi=150, bbox_inches="tight", facecolor="white",
    )
    fig.savefig(
        os.path.join(save_dir, f"{safe}.pdf"),
        bbox_inches="tight", facecolor="white",
    )
    plt.close(fig)


def plot_paper_figures(results_dir: str, save_dir: str) -> None:
    """Generate publication-quality compression–quality figures.

    Loads all ``*_vtcb.json`` files from ``results_dir`` and writes:

    - One PNG + PDF per metric (e.g. ``radgraph_f1.png``, ``radgraph_f1.pdf``)
    - ``summary_grid.png`` and ``summary_grid.pdf`` — 2×2 grid of the four
      primary metrics: RadGraph-F1, RaTEScore, AUROC (macro), Recall@5

    Args:
        results_dir: Directory containing ``*_vtcb.json`` result files.
        save_dir:    Output directory for figures.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    loaded = _load_results(results_dir)
    all_metrics = _collect_metrics(loaded)

    # ── Per-metric figures ─────────────────────────────────────────────────────
    for metric_name in all_metrics:
        label = _METRIC_LABELS.get(
            metric_name, metric_name.replace("_", " ").title()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor("white")
        _plot_metric_on_ax(ax, loaded, metric_name)
        _apply_paper_style(ax, "Token budget M", label, f"{label} vs Token Budget")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels,
                fontsize=8, loc="best",
                framealpha=0.9, edgecolor="0.8",
            )
        fig.tight_layout()
        _save(fig, save_dir, metric_name)

    # ── 2×2 summary grid ──────────────────────────────────────────────────────
    primary = [m for m in _PRIMARY_METRICS if m in all_metrics]
    # Pad with remaining metrics if fewer than 4 primary ones are present
    for m in all_metrics:
        if m not in primary and len(primary) < 4:
            primary.append(m)
    while len(primary) < 4:
        primary.append(None)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.patch.set_facecolor("white")

    for ax, metric_name in zip(axes.flat, primary):
        if metric_name is None:
            ax.axis("off")
            continue
        label = _METRIC_LABELS.get(
            metric_name, metric_name.replace("_", " ").title()
        )
        _plot_metric_on_ax(ax, loaded, metric_name)
        _apply_paper_style(ax, "Token budget M", label, label)
        handles, labels_ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles, labels_,
                fontsize=7, loc="best",
                framealpha=0.9, edgecolor="0.8",
            )

    fig.suptitle("Compression–Quality Tradeoff", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, save_dir, "summary_grid")
