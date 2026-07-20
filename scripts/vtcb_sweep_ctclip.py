#!/usr/bin/env python3
"""VTCB token-budget sweep for a trained CTCLIPStage2VLM checkpoint.

For each M in vtcb_budgets:
  1. Load the trained model from checkpoint_best.pt.
  2. Rebuild Stage 2 at budget M (fresh head, no weight copying).
  3. Fine-tune for 1 epoch on train set at lr=1e-5 (100-step warm-up).
  4. Generate reports for all val volumes and score with text metrics.
  5. Save results to results/vtcb_sweep_ctclip/M={M}_scores.json.
  6. Print summary table.

Usage
-----
    python scripts/vtcb_sweep_ctclip.py \\
        --config configs/ctclip_stage2.yaml \\
        --checkpoint checkpoints/ctclip_stage2/checkpoint_best.pt

Override budgets::

    python scripts/vtcb_sweep_ctclip.py \\
        --config configs/ctclip_stage2.yaml \\
        --checkpoint checkpoints/ctclip_stage2/checkpoint_best.pt \\
        --budgets 16 32 64 128
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────


def _load_config(path: str) -> Dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


# ── Text metrics ───────────────────────────────────────────────────────────────


def _score_generations(predictions: List[str], references: List[str]) -> Dict:
    """Compute BLEU-1, BLEU-4, METEOR, ROUGE-L, CIDEr against references.

    Falls back gracefully if a metric library is unavailable.
    """
    scores: Dict[str, float] = {}

    # BLEU + METEOR via nltk
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score
        import nltk
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("wordnet", quiet=True)

        smoothing = SmoothingFunction().method1
        refs_tok = [[r.split()] for r in references]
        hyps_tok = [p.split() for p in predictions]

        scores["bleu_1"] = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0),
                                        smoothing_function=smoothing)
        scores["bleu_4"] = corpus_bleu(refs_tok, hyps_tok,
                                        smoothing_function=smoothing)
        scores["meteor"] = float(
            sum(meteor_score([r.split()], p.split())
                for r, p in zip(references, predictions))
            / max(len(predictions), 1)
        )
    except Exception as exc:
        logger.warning("BLEU/METEOR scoring failed: %s", exc)

    # ROUGE-L via rouge_score
    try:
        from rouge_score import rouge_scorer as rs_module
        scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)
        rouge_scores = [
            scorer.score(ref, pred)["rougeL"].fmeasure
            for ref, pred in zip(references, predictions)
        ]
        scores["rouge_l"] = float(sum(rouge_scores) / max(len(rouge_scores), 1))
    except Exception as exc:
        logger.warning("ROUGE-L scoring failed: %s", exc)

    # RadGraph F1 via aadp metric (may not be available in all environments)
    try:
        from aadp.evaluation.metrics.radgraph_f1 import compute_radgraph_f1
        rg_scores = compute_radgraph_f1(predictions, references)
        scores["radgraph_f1"] = rg_scores.get("f1", float("nan"))
    except Exception as exc:
        logger.debug("RadGraph F1 unavailable: %s", exc)

    return scores


# ── Short fine-tune at new budget ──────────────────────────────────────────────


def _finetune_one_epoch(
    model,
    train_loader: DataLoader,
    device: str,
    lr: float = 1e-5,
    warmup_steps: int = 100,
    max_steps: Optional[int] = None,
) -> None:
    from aadp.training.scheduler import get_cosine_schedule_with_warmup

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)
    total_steps = max_steps or len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    model.train()
    for step, batch in enumerate(train_loader):
        if max_steps is not None and step >= max_steps:
            break
        features = batch["features"].to(device)
        instructions = batch["instruction"]
        target_enc = model.tokenizer(
            batch["target"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).input_ids.to(device)

        out = model(features, instructions, report_tokens=target_enc, training=True)
        loss = out["loss"]
        loss.backward()
        clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % 20 == 0:
            logger.info("  finetune step=%d  loss=%.4f", step, loss.item())

    model.eval()


# ── Generation + scoring for one budget ───────────────────────────────────────


@torch.no_grad()
def _evaluate_budget(model, val_loader: DataLoader, device: str) -> Dict:
    model.eval()
    predictions: List[str] = []
    references: List[str] = []

    for batch in val_loader:
        features = batch["features"].to(device)
        # Use T1 instruction for report generation eval
        instructions = ["Generate a radiology report for this CT scan."] * len(features)
        out = model(features, instructions, training=False)
        gen_ids = out["generated_ids"]

        decoded = model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend(batch["target"])

    scores = _score_generations(predictions, references)
    scores["n_samples"] = len(predictions)
    return scores


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="VTCB token-budget sweep for CT-CLIP Stage 2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_best.pt")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Override vtcb_budgets from config")
    parser.add_argument("--finetune_steps", type=int, default=None,
                        help="Max fine-tune steps per budget (default: one full epoch)")
    parser.add_argument("--results_dir", type=str, default="results/vtcb_sweep_ctclip")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    budgets: List[int] = args.budgets or cfg.get("vtcb_budgets", [16, 32, 64, 128, 256, 512])
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    from aadp.data.ctclip_feature_dataset import CTCLIPFeatureDataset, ctclip_collate_fn

    train_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_train_dir"],
        csv_path=cfg["ctrate_csv_train"],
        tasks=cfg.get("tasks", ["T1", "T2", "T3"]),
        task_weights=cfg.get("task_weights", {"T1": 0.6, "T2": 0.3, "T3": 0.1}),
    )
    val_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_valid_dir"],
        csv_path=cfg["ctrate_csv_valid"],
        tasks=["T1"],  # T1 only for generation eval
    )

    batch_size = cfg.get("batch_size", 8)
    num_workers = min(cfg.get("num_workers", 4), os.cpu_count() or 2)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, collate_fn=ctclip_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, collate_fn=ctclip_collate_fn)

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = torch.load(args.checkpoint, map_location=device)

    all_results: Dict[int, Dict] = {}

    for M in budgets:
        logger.info("=" * 60)
        logger.info("Budget M=%d — rebuilding Stage 2...", M)

        # Fresh model for each budget (avoid state leaking between runs)
        from aadp.models.ctclip_vlm import CTCLIPStage2VLM

        model = CTCLIPStage2VLM(
            ctclip_dim=cfg.get("ctclip_dim", 512),
            embed_dim=cfg.get("embed_dim", 512),
            num_tokens=cfg.get("num_tokens", 64),
            num_heads=cfg.get("num_heads", 8),
            cond_dim=cfg.get("cond_dim", 2048),
            use_film=cfg.get("use_film", True),
            max_depth=cfg.get("max_depth", 24),
            dropout=cfg.get("dropout", 0.0),
            llm_model_name=cfg.get("llm_model_name", "facebook/opt-1.3b"),
            llm_frozen=cfg.get("llm_frozen", False),
            llm_lora=cfg.get("llm_lora"),
            instruction_encoder_model=cfg.get("instruction_encoder_model", "facebook/opt-1.3b"),
            device=device,
        )

        # Restore trained projector/visual_proj weights, then rebuild Stage 2 at M
        model.projector.load_state_dict(ckpt["projector"])
        model.visual_proj.load_state_dict(ckpt["visual_proj"])
        if "llm_lora" in ckpt:
            model.llm.load_state_dict(ckpt["llm_lora"], strict=False)

        model.rebuild_at_budget(M)
        logger.info("Stage 2 rebuilt at M=%d. Fine-tuning for 1 epoch (lr=1e-5)...", M)

        _finetune_one_epoch(
            model, train_loader, device,
            lr=1e-5, warmup_steps=100,
            max_steps=args.finetune_steps,
        )

        logger.info("Evaluating M=%d on val set...", M)
        scores = _evaluate_budget(model, val_loader, device)
        all_results[M] = scores

        # Save per-budget result
        result_path = results_dir / f"M={M}_scores.json"
        with open(result_path, "w") as fh:
            json.dump({"M": M, "scores": scores}, fh, indent=2)
        logger.info("Results saved → %s", result_path)

        # Free memory before next budget
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Summary ───────────────────────────────────────────────────────────────
    full_results_path = results_dir / "vtcb_sweep_summary.json"
    with open(full_results_path, "w") as fh:
        json.dump({"budgets": budgets, "results": {str(k): v for k, v in all_results.items()}}, fh, indent=2)
    logger.info("Full results saved → %s", full_results_path)

    # Print summary table
    metrics = [k for k in next(iter(all_results.values())).keys() if k != "n_samples"]
    header = f"{'M':>6}  " + "  ".join(f"{m:>12}" for m in metrics)
    print("\n" + "=" * len(header))
    print("VTCB Sweep Summary")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for M in budgets:
        row = all_results.get(M, {})
        vals = "  ".join(f"{row.get(m, float('nan')):>12.4f}" for m in metrics)
        print(f"{M:>6}  {vals}")
    print("=" * len(header))


if __name__ == "__main__":
    main()
