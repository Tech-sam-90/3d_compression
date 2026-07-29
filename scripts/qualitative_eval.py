#!/usr/bin/env python3
"""qualitative_eval.py — standalone qualitative check of a trained ICTC checkpoint.

For a handful of validation scans, runs THREE different instructions on the
SAME scan and prints (generated, ground truth) side by side, to demonstrate
whether the conditioning mechanism (FiLM or attention-conditioned, whichever
the checkpoint was trained with — read from --config) actually changes what
the model focuses on, or whether it collapses to the same boilerplate
regardless of instruction.

Ground truth: Findings_EN + " " + Impressions_EN from the report CSV — the
same concatenation CTCLIPFeatureDataset.__getitem__ uses to build the T1
("Generate a radiology report...") target (aadp/data/ctclip_feature_dataset.py).

Read-only / inference-only: loads the checkpoint, runs generation, prints
results. No training, no code modification, no checkpoint writes.

Usage:
    python scripts/qualitative_eval.py \\
        --ckpt /path/to/checkpoint.pt \\
        --config configs/ctclip_stage2_final.yaml \\
        --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid \\
        --report_csv /path/to/valid_merged.csv
"""
import argparse
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import yaml

# ── Fixed instruction set ────────────────────────────────────────────────────

INSTRUCTION_A = "Generate a detailed radiology report describing all findings."
INSTRUCTION_B = (
    "Describe only pulmonary findings including airways, lung parenchyma, "
    "and pleural spaces."
)
INSTRUCTION_C = "Describe cardiac and mediastinal structures only."

# Scan selection per the task: 2 from the "collapsed" cluster identified in
# docs/DIAGNOSTIC_REPORT.md (recurred across multiple checkpoints' Test 3
# reruns) + 3 from the "distinct" group (stayed differentiated everywhere).
DEFAULT_SCANS = [
    "valid_131_a_1",   # collapsed
    "valid_634_a_1",   # collapsed
    "valid_1_a_1",      # distinct
    "valid_382_c_2",    # distinct
    "valid_500_d_1",    # distinct
]

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512


# ── Bag-of-words cosine similarity (no new installs: stdlib only) ───────────

def bow_cosine(a: str, b: str) -> float:
    """Cosine similarity between two texts' word-count vectors.

    Deliberately avoids sentence_transformers/sklearn — neither is a
    guaranteed dependency of this project's env, and a plain bag-of-words
    cosine (stdlib re/collections/math only) is sufficient to detect gross
    "outputs are near-identical" vs. "outputs diverge" collapse, which is
    all this diagnostic needs.
    """
    wa = re.findall(r"\w+", a.lower())
    wb = re.findall(r"\w+", b.lower())
    ca, cb = Counter(wa), Counter(wb)
    if not ca or not cb:
        return 0.0
    shared = set(ca) & set(cb)
    dot = sum(ca[w] * cb[w] for w in shared)
    norm_a = math.sqrt(sum(v * v for v in ca.values()))
    norm_b = math.sqrt(sum(v * v for v in cb.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_feature(feature_dir: Path, stem: str) -> torch.Tensor:
    pt_path = feature_dir / f"{stem}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No feature file for {stem!r} at {pt_path}")
    feat = torch.load(pt_path, weights_only=True).float()
    return feat.reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)


def load_ground_truth(report_csv: str, stem: str) -> str:
    """Same convention as CTCLIPFeatureDataset.__getitem__: Findings_EN + ' ' + Impressions_EN."""
    df = pd.read_csv(report_csv)
    df["_stem"] = df["VolumeName"].astype(str).apply(
        lambda v: v[:-7] if v.endswith(".nii.gz") else Path(v).stem
    )
    row = df[df["_stem"] == stem]
    if len(row) == 0:
        return "[NOT FOUND IN report_csv]"
    row = row.iloc[0]
    findings = str(row.get("Findings_EN") or "").strip()
    impressions = str(row.get("Impressions_EN") or "").strip()
    return (findings + " " + impressions).strip()


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, features_1: torch.Tensor, instruction: str, max_new_tokens: int) -> str:
    """Replicates CTCLIPStage2VLM.forward()'s inference path (instruction
    encoder -> projector -> visual_proj -> concat with tokenized instruction
    -> llm.generate()) without modifying forward() itself. Greedy decoding
    (do_sample=False, num_beams=1) for reproducibility."""
    device = features_1.device
    etext = model.instruction_encoder([instruction]).float()
    visual = model.projector(features_1, etext)
    visual = model.visual_proj(visual).to(model.llm.dtype)

    inst_enc = model.tokenizer(
        [instruction], return_tensors="pt", padding=True, truncation=True, max_length=128
    ).to(device)
    embed_fn = model.llm.get_input_embeddings()
    inst_embeds = embed_fn(inst_enc.input_ids).to(model.llm.dtype)

    inputs_embeds = torch.cat([visual, inst_embeds], dim=1)
    B, L, _ = inputs_embeds.shape
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)

    gen_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=model.tokenizer.eos_token_id,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text.strip()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint_best.pt (or any checkpoint file)")
    parser.add_argument("--config", required=True, help="Path to the training config YAML (architecture/LLM must match --ckpt)")
    parser.add_argument("--feature_dir", default="/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
    parser.add_argument("--report_csv", default=None,
                         help="CT-RATE reports CSV with VolumeName/Findings_EN/Impressions_EN. "
                              "Defaults to the config's ctrate_csv_valid if not given.")
    parser.add_argument("--scans", default=None,
                         help="Comma-separated scan stems to evaluate. Defaults to the 5 scans "
                              "from docs/DIAGNOSTIC_REPORT.md's collapsed/distinct sets.")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    report_csv = args.report_csv or cfg["ctrate_csv_valid"]
    scans = args.scans.split(",") if args.scans else DEFAULT_SCANS
    feature_dir = Path(args.feature_dir)
    device = args.device

    print(f"Loading model (llm={cfg.get('llm_model_name')}, conditioning={cfg.get('aggregator_conditioning', 'film')}) "
          f"on {device}, bfloat16...")

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
        top_k=cfg.get("aggregator_top_k", 128),
        conditioning=cfg.get("aggregator_conditioning", "film"),
        llm_model_name=cfg.get("llm_model_name", "facebook/opt-1.3b"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model", "facebook/opt-1.3b"),
        device=device,
    )

    ckpt = torch.load(args.ckpt, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    print(f"Checkpoint loaded: step={ckpt.get('step')} epoch={ckpt.get('epoch')} "
          f"val_loss={ckpt.get('val_loss', float('nan')):.4f}")

    # Inference-only: safe to cast the whole model (frozen instruction
    # encoder, trained aggregator, trained LoRA-wrapped LLM) to bfloat16 —
    # no backward pass here, so none of ctclip_vlm.py's fp32-for-training-
    # stability reasoning applies. bf16 (not fp16) avoids the narrow
    # exponent range that causes overflow instability in LLM inference.
    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    results = {}  # stem -> {"A": text, "B": text, "C": text, "gt": text}

    for stem in scans:
        print("\n" + "─" * 70)
        print(f"SCAN: {stem}")
        print("─" * 70)

        try:
            feat = load_feature(feature_dir, stem).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            continue

        gt = load_ground_truth(report_csv, stem)

        gen_a = generate(model, feat, INSTRUCTION_A, args.max_new_tokens)
        print("INSTRUCTION A — Full report:")
        print(f"GENERATED : {gen_a}")
        print(f"GROUND TRUTH: {gt}")
        print()

        gen_b = generate(model, feat, INSTRUCTION_B, args.max_new_tokens)
        print("INSTRUCTION B — Pulmonary only:")
        print(f"GENERATED : {gen_b}")
        print(f"GROUND TRUTH: {gt}")
        print()

        gen_c = generate(model, feat, INSTRUCTION_C, args.max_new_tokens)
        print("INSTRUCTION C — Cardiac/mediastinal only:")
        print(f"GENERATED : {gen_c}")
        print(f"GROUND TRUTH: {gt}")
        print("─" * 70)

        results[stem] = {"A": gen_a, "B": gen_b, "C": gen_c, "gt": gt}

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY — cosine(INSTRUCTION A output, INSTRUCTION B output)")
    print("=" * 70)

    cosines = []
    for stem, r in results.items():
        cos_ab = bow_cosine(r["A"], r["B"])
        cosines.append(cos_ab)
        print(f"  {stem}: cosine(A, B) = {cos_ab:.4f}")

    if cosines:
        n = len(cosines)
        n_low = sum(1 for c in cosines if c < 0.85)
        n_high = sum(1 for c in cosines if c > 0.95)
        mean_cos = sum(cosines) / n
        print(f"\nMean cosine(A, B) across {n} scans: {mean_cos:.4f}")

        if n_low > n / 2:
            print("CONDITIONING WORKING — outputs diverge with different instructions")
        elif n_high > n / 2:
            print("WARNING — outputs too similar, conditioning may not be steering content")
        else:
            print("MIXED — neither clearly diverging nor clearly collapsed; inspect per-scan reports above")
    else:
        print("No scans evaluated — nothing to summarize.")


if __name__ == "__main__":
    main()
