#!/usr/bin/env python3
"""evaluate_checkpoint.py — direct evaluation of a trained CTCLIPStage2VLM
checkpoint at its native architecture (no budget rebuild, no extra
fine-tuning).

Unlike scripts/vtcb_sweep_ctclip.py (which always rebuilds Stage 2 at a NEW
token budget and fine-tunes 1 epoch before evaluating — none of its 6 runs
directly evaluates what training actually produced), this script loads the
checkpoint as-is at the config's native num_tokens and evaluates it
directly. Also unlike scripts/evaluate.py (which builds the wrong model
class — MedVLM, the full ViT+Stage1+Stage2+LLM pipeline this branch does
NOT use — not CTCLIPStage2VLM), this uses the correct model.

Prints a handful of (prediction, reference) pairs for qualitative
inspection, then the full compute_all_metrics() scores over the sample.

Usage:
    python scripts/evaluate_checkpoint.py \\
        --config configs/ctclip_stage2.yaml \\
        --checkpoint /scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt \\
        --max_samples 200 \\
        --n_examples 5
"""

import argparse
import json
import logging

import torch
import yaml
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max_samples", type=int, default=200,
                         help="Cap on validation samples scored (None = full val set)")
    parser.add_argument("--n_examples", type=int, default=5,
                         help="Number of (prediction, reference) pairs to print")
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA unavailable — falling back to CPU.")
        device = "cpu"

    from aadp.data.ctclip_feature_dataset import CTCLIPFeatureDataset, ctclip_collate_fn

    logger.info("Loading validation dataset from %s", cfg["features_valid_dir"])
    val_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_valid_dir"],
        csv_path=cfg["ctrate_csv_valid"],
        tasks=["T1"],  # report generation only
        max_samples=args.max_samples,
    )
    batch_size = args.batch_size or cfg.get("batch_size", 8)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=ctclip_collate_fn,
    )
    logger.info("Evaluating on %d validation samples.", len(val_ds))

    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    logger.info("Building CTCLIPStage2VLM at native num_tokens=%s (llm=%s)...",
                cfg.get("num_tokens", 64), cfg.get("llm_model_name"))
    model = CTCLIPStage2VLM(
        ctclip_dim=cfg.get("ctclip_dim", 512),
        embed_dim=cfg.get("embed_dim", 512),
        num_tokens=cfg.get("num_tokens", 64),
        num_heads=cfg.get("num_heads", 8),
        cond_dim=cfg.get("cond_dim", 2048),
        use_film=cfg.get("use_film", True),
        max_depth=cfg.get("max_depth", 24),
        dropout=cfg.get("dropout", 0.0),
        top_k=cfg.get("aggregator_top_k", 128),
        conditioning=cfg.get("aggregator_conditioning", "film"),
        llm_model_name=cfg.get("llm_model_name", "facebook/opt-1.3b"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model", "facebook/opt-1.3b"),
        device=device,
    )

    logger.info("Loading checkpoint weights from %s", args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    logger.info(
        "Checkpoint loaded (step=%s, epoch=%s, val_loss=%.4f).",
        ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss", float("nan")),
    )

    model.eval()
    predictions, references = [], []
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            instructions = ["Generate a radiology report for this CT scan."] * len(features)
            out = model(features, instructions, training=False)
            decoded = model.tokenizer.batch_decode(out["generated_ids"], skip_special_tokens=True)
            predictions.extend(decoded)
            references.extend(batch["target"])
            logger.info("Generated %d/%d", len(predictions), len(val_ds))

    print("\n" + "=" * 80)
    print(f"QUALITATIVE EXAMPLES (first {args.n_examples})")
    print("=" * 80)
    for i in range(min(args.n_examples, len(predictions))):
        print(f"\n--- Example {i} ---")
        print(f"REFERENCE:  {references[i][:500]}")
        print(f"PREDICTION: {predictions[i][:500]}")

    from aadp.evaluation.metrics.compute_all import compute_all_metrics

    print("\n" + "=" * 80)
    print(f"METRICS (n={len(predictions)} samples)")
    print("=" * 80)
    scores = compute_all_metrics(predictions, references)
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
