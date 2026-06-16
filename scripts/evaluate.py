#!/usr/bin/env python3
"""CLI for running the VTCB benchmark or comparing/plotting results.

Usage
-----
Run evaluation on a trained checkpoint::

    python scripts/evaluate.py \\
        --config configs/aadp_base.yaml \\
        --checkpoint checkpoints/checkpoint_best.pt

Sweep specific token budgets::

    python scripts/evaluate.py \\
        --config configs/aadp_base.yaml \\
        --checkpoint checkpoints/checkpoint_best.pt \\
        --budgets 16 32 64 128

Compare two result JSONs::

    python scripts/evaluate.py \\
        --compare results/aadp_vtcb.json results/perceiver_vtcb.json \\
        --model_names A-ADP Perceiver

Plot compression curves::

    python scripts/evaluate.py \\
        --plot results/aadp_vtcb.json results/perceiver_vtcb.json \\
        --model_names A-ADP Perceiver \\
        --metrics radgraph_f1 ratescore_mean
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_config(config_path: str) -> dict:
    with open(config_path) as fh:
        return yaml.safe_load(fh)


def _build_model(config: dict, device: str):
    from aadp.models.vlm import MedVLM

    return MedVLM(
        vit_model_name=config.get("vit_model_name", "vit_base_patch16_224"),
        vit_pretrained=False,
        llm_model_name=config.get("llm_model_name", "facebook/opt-125m"),
        num_latents=config.get("num_latents", 32),
        num_tokens=config.get("num_tokens", 64),
        use_film=config.get("use_film", True),
        max_depth=config.get("max_depth", 512),
        instruction_encoder_model=config.get(
            "instruction_encoder_model",
            config.get("llm_model_name", "facebook/opt-125m"),
        ),
        device=device,
    )


def _load_checkpoint(model, checkpoint_path: str, device: str) -> None:
    if not os.path.exists(checkpoint_path):
        logger.warning("Checkpoint not found at %s — skipping weight load.", checkpoint_path)
        return
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "projector" in ckpt:
        model.projector.load_state_dict(ckpt["projector"])
        logger.info("Loaded projector weights from %s", checkpoint_path)
    elif "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        logger.info("Loaded full model weights from %s", checkpoint_path)
    else:
        logger.warning(
            "Unknown checkpoint format (keys: %s) — skipping.", list(ckpt.keys())
        )


def _build_dataset(config: dict):
    try:
        from aadp.data.ctratedataset import CTRATEDataset
        from dotenv import load_dotenv

        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        split = config.get("val_split", "validation")
        return CTRATEDataset(split=split, hf_token=hf_token)
    except Exception as exc:
        logger.warning("Could not load CTRATEDataset (%s) — using empty dataset.", exc)
        return []


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VTCB benchmark runner / compare / plot")

    p.add_argument("--config", type=str, help="Path to YAML training config")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to model checkpoint (.pt)")
    p.add_argument("--budgets", type=int, nargs="+", default=None,
                   help="Token budgets to sweep (overrides config default)")
    p.add_argument("--primary_budget", type=int, default=64)
    p.add_argument("--model_name", type=str, default=None,
                   help="Name written into the result JSON filename")
    p.add_argument("--results_dir", type=str, default="results/")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap on validation samples")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--compare", type=str, nargs="+", default=None,
                   help="Result JSON files to compare")
    p.add_argument("--model_names", type=str, nargs="+", default=None,
                   help="Display names for --compare / --plot models")

    p.add_argument("--plot", type=str, nargs="+", default=None,
                   help="Result JSON files to plot compression curves for")
    p.add_argument("--metrics", type=str, nargs="+",
                   default=["radgraph_f1", "ratescore_mean"])
    p.add_argument("--plot_dir", type=str, default=None)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Compare mode ──────────────────────────────────────────────────────────
    if args.compare is not None:
        from aadp.evaluation.benchmarks.vtcb import VTCBRunner

        names = args.model_names or [Path(p).stem for p in args.compare]
        comparison = VTCBRunner.compare(dict(zip(names, args.compare)))
        print(json.dumps(comparison, indent=2))
        return

    # ── Plot mode ─────────────────────────────────────────────────────────────
    if args.plot is not None:
        from aadp.evaluation.benchmarks.vtcb import VTCBRunner

        names = args.model_names or [Path(p).stem for p in args.plot]
        plot_dir = args.plot_dir or os.path.join(args.results_dir, "plots")
        VTCBRunner.plot_compression_curves(dict(zip(names, args.plot)), args.metrics, plot_dir)
        logger.info("Plots saved to %s", plot_dir)
        return

    # ── Evaluation mode ───────────────────────────────────────────────────────
    if args.config is None:
        print("Error: --config is required for evaluation mode.", file=sys.stderr)
        sys.exit(1)

    config = _load_config(args.config)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    model = _build_model(config, device)

    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint, device)

    val_dataset = _build_dataset(config)

    budgets = args.budgets or config.get("token_budgets", [16, 32, 64, 128])
    model_name = args.model_name or config.get("experiment_name", "model")
    batch_size = args.batch_size or config.get("batch_size", 4)

    from aadp.evaluation.benchmarks.vtcb import VTCBRunner

    runner = VTCBRunner(
        model=model,
        val_dataset=val_dataset,
        token_budgets=budgets,
        primary_budget=args.primary_budget,
        batch_size=batch_size,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        device=device,
        results_dir=args.results_dir,
    )

    results = runner.run(model_name=model_name)
    logger.info(
        "VTCB complete. Task families: %s",
        {str(M): [k for k in v if not k.startswith("_")] for M, v in results.items()},
    )


if __name__ == "__main__":
    main()
