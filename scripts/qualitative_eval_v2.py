#!/usr/bin/env python3
"""qualitative_eval_v2.py — qualitative check using ACTUAL training templates.

qualitative_eval.py (v1) used three hand-written instruction paraphrases
("Generate a detailed radiology report describing all findings.", "Describe
only pulmonary findings...", "Describe cardiac and mediastinal structures
only.") that the model never saw during training. That turned out to matter:
v1 found severe cross-scan collapse on Instructions B/C, but Test 3 (which
uses the EXACT training-time T1 string) found 0% collapse on the same
checkpoint and scans. This script replaces the hand-written paraphrases with
build_instruction_for_task()'s actual T1/T2/T3 template logic
(aadp/data/ctclip_feature_dataset.py) — the exact wording
scripts/train_ctclip.py feeds the model during training (tasks=["T1","T2","T3"],
weights 60/30/10, every config this session) — so this test asks: does the
model collapse on instructions it was actually trained on, or only on
out-of-distribution paraphrases?

T1/T2 reuse aadp.data.instruction_builder.extract_entities_from_report /
sentences_containing directly (same functions ctclip_feature_dataset.py
imports) rather than reimplementing them. T3's abnormality labels come from
the report CSV's binary label columns (the same 18 columns
CTCLIPFeatureDataset auto-detects) — NOT from the .pt feature files, which
contain only visual features, no labels; T3a/T3b pick the first positive/
negative label column (sorted order) for reproducibility, unlike training's
random.choice.

Writes a full markdown report to --out in addition to printing progress to
stdout (so the SLURM log stays a live trace while the .md file is the actual
deliverable).

Read-only / inference-only: loads the checkpoint, runs generation, writes a
report. No training, no code modification, no checkpoint writes.

Usage:
    python scripts/qualitative_eval_v2.py \\
        --ckpt /scratch/sadeniji/ictc_checkpoints_vfilm_fix/checkpoint_best.pt \\
        --config configs/ctclip_vfilm_fix.yaml \\
        --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid \\
        --out docs/QUALITATIVE_EVAL_V2.md
"""
import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import yaml

from aadp.data.instruction_builder import extract_entities_from_report, sentences_containing

DEFAULT_SCANS = [
    "valid_131_a_1",
    "valid_634_a_1",
    "valid_1_a_1",
    "valid_382_c_2",
    "valid_500_d_1",
]

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512

# Same exclusion set as aadp/data/ctclip_feature_dataset.py's _NON_LABEL_COLS.
# Duplicated rather than imported (it's a private module constant and this is
# a standalone diagnostic script) so T3 detects the same 18 label columns
# training actually samples abnormalities from.
_NON_LABEL_COLS = {
    "VolumeName", "ClinicalInformation_EN", "Technique_EN",
    "Findings_EN", "Impressions_EN",
    "split", "Split", "PatientID", "StudyDate",
}

FALLBACK_T3A = "Is there evidence of consolidation in this scan? Answer yes or no."
FALLBACK_T3B = "Is there evidence of pneumothorax in this scan? Answer yes or no."

NEAR_DUP_THRESHOLD = 0.05  # normalized edit distance — same threshold Test 3 uses


# ── Bag-of-words cosine similarity (stdlib only — see qualitative_eval.py) ──

def bow_cosine(a: str, b: str) -> float:
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


def edit_distance(a: str, b: str) -> int:
    """Same implementation as scripts/diagnostics/test3_full_rerun_*.py, for
    a consistent collapse-detection threshold across this project's
    diagnostics."""
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def count_distinct(outputs: List[str]) -> int:
    """Count how many of ``outputs`` have no near-duplicate partner
    (normalized edit distance < NEAR_DUP_THRESHOLD) among the rest."""
    n = len(outputs)
    distinct = 0
    for i in range(n):
        has_dup = False
        for j in range(n):
            if i == j:
                continue
            max_len = max(len(outputs[i]), len(outputs[j]), 1)
            if edit_distance(outputs[i], outputs[j]) / max_len < NEAR_DUP_THRESHOLD:
                has_dup = True
                break
        if not has_dup:
            distinct += 1
    return distinct


# ── Data loading ──────────────────────────────────────────────────────────────

def load_feature(feature_dir: Path, stem: str) -> torch.Tensor:
    pt_path = feature_dir / f"{stem}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No feature file for {stem!r} at {pt_path}")
    feat = torch.load(pt_path, weights_only=True).float()
    return feat.reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)


def detect_label_cols(df: pd.DataFrame) -> List[str]:
    """Mirrors CTCLIPFeatureDataset.__init__'s label-column detection
    (aadp/data/ctclip_feature_dataset.py) so T3 uses the same 18 columns
    training actually samples abnormalities from."""
    candidate = [
        c for c in df.columns
        if c not in _NON_LABEL_COLS and pd.api.types.is_integer_dtype(df[c])
    ]
    for c in df.columns:
        if c not in _NON_LABEL_COLS and c not in candidate:
            try:
                unique_vals = df[c].dropna().unique()
                if len(unique_vals) > 0 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    candidate.append(c)
            except Exception:
                pass
    return sorted(candidate)


def load_row(df: pd.DataFrame, stem: str) -> Optional[pd.Series]:
    row = df[df["_stem"] == stem]
    return row.iloc[0] if len(row) else None


def report_from_row(row: pd.Series) -> str:
    findings = str(row.get("Findings_EN") or "").strip()
    impressions = str(row.get("Impressions_EN") or "").strip()
    return (findings + " " + impressions).strip()


def label_dict_from_row(row: pd.Series, label_cols: List[str]) -> Dict[str, int]:
    d = {}
    for col in label_cols:
        val = row.get(col, 0)
        try:
            d[col] = int(val) if not pd.isna(val) else 0
        except Exception:
            d[col] = 0
    return d


def build_t3_pairs(label_dict: Dict[str, int]):
    """First positive-labelled and first negative-labelled abnormality
    (sorted-column order — deterministic, unlike training's random.choice,
    so this diagnostic is reproducible run to run). Falls back to a fixed
    question when no label of that polarity exists for this scan (expected
    answer becomes unknown — accuracy is not scored for that fallback)."""
    positives = [k for k, v in label_dict.items() if v == 1]
    negatives = [k for k, v in label_dict.items() if v == 0]

    if positives:
        abn = positives[0].strip().lower()
        t3a = (f"Is there evidence of {abn} in this scan? Answer yes or no.", abn, True)
    else:
        t3a = (FALLBACK_T3A, "consolidation", None)  # expected unknown -> not scored

    if negatives:
        abn = negatives[0].strip().lower()
        t3b = (f"Is there evidence of {abn} in this scan? Answer yes or no.", abn, False)
    else:
        t3b = (FALLBACK_T3B, "pneumothorax", None)

    return t3a, t3b


def check_yes_no(generated: str, expected_yes: bool) -> Optional[bool]:
    """First yes/no token in the generated text vs. the expected answer."""
    m = re.search(r"\b(yes|no)\b", generated.lower())
    if not m:
        return False  # no clear answer given -> counted wrong, not skipped
    said_yes = m.group(1) == "yes"
    return said_yes == expected_yes


# ── Generation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, features_1: torch.Tensor, instruction: str, max_new_tokens: int) -> str:
    """Same inference path as qualitative_eval.py's generate()."""
    device = features_1.device
    etext = model.instruction_encoder([instruction]).to(model.llm.dtype)
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
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature_dir", default="/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
    parser.add_argument("--report_csv", default=None)
    parser.add_argument("--scans", default=None)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="docs/QUALITATIVE_EVAL_V2.md")
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
    step, val_loss = ckpt.get("step"), ckpt.get("val_loss", float("nan"))
    print(f"Checkpoint loaded: step={step} epoch={ckpt.get('epoch')} val_loss={val_loss:.4f}")

    model = model.to(device=device, dtype=torch.bfloat16)
    model.eval()

    print(f"Loading report CSV from {report_csv}...")
    df = pd.read_csv(report_csv)
    df["_stem"] = df["VolumeName"].astype(str).apply(
        lambda v: v[:-7] if v.endswith(".nii.gz") else Path(v).stem
    )
    label_cols = detect_label_cols(df)
    print(f"Detected {len(label_cols)} label columns.")

    md: List[str] = []
    md.append("# Qualitative Evaluation V2 — V-FiLM Fix Checkpoint")
    md.append(f"(training job 66667523, step={step}, val_loss={val_loss:.4f})")
    md.append("")
    md.append("## Key change from V1")
    md.append(
        "V1 (`scripts/qualitative_eval.py`) used three hand-written instruction "
        "paraphrases the model never saw during training. This version replaces "
        "them with `build_instruction_for_task()`'s actual T1/T2/T3 template "
        "logic (`aadp/data/ctclip_feature_dataset.py`) — the exact instruction "
        "wording every training run this session (including this checkpoint's, "
        "job 66667523) actually samples from at 60/30/10 weights "
        "(`scripts/train_ctclip.py:300-314`). T1 is the fixed training string; "
        "T2a/T2b use `extract_entities_from_report`/`sentences_containing` on "
        "each scan's own ground-truth report (same functions "
        "`ctclip_feature_dataset.py` imports); T3a/T3b use the first positive/"
        "negative label column from the report CSV's 18 binary abnormality "
        "columns (deterministic, unlike training's `random.choice`, for "
        "reproducibility)."
    )
    md.append("")
    md.append("## Per-scan results")
    md.append("")

    t1_outputs, t2_lung_outputs = [], {}
    t3_correct, t3_scored = 0, 0

    for stem in scans:
        print(f"\n--- SCAN: {stem} ---")
        try:
            feat = load_feature(feature_dir, stem).unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")
            md.append(f"### SCAN: {stem}\n\nSKIPPED: {e}\n")
            continue

        row = load_row(df, stem)
        if row is None:
            print(f"  SKIPPED: {stem!r} not found in {report_csv}")
            md.append(f"### SCAN: {stem}\n\nSKIPPED: not found in report CSV\n")
            continue

        gt_report = report_from_row(row)
        label_dict = label_dict_from_row(row, label_cols)

        md.append(f"### SCAN: {stem}")
        md.append("")

        # ── T1 ────────────────────────────────────────────────────────────
        instr_t1 = "Generate a radiology report for this CT scan."
        gen_t1 = generate(model, feat, instr_t1, args.max_new_tokens)
        t1_outputs.append(gen_t1)
        print(f"  T1: {gen_t1[:100]}...")
        md += [
            "**T1 — Full report**",
            f'- Instruction: "{instr_t1}"',
            f"- GENERATED: {gen_t1}",
            f"- GROUND TRUTH: {gt_report}",
            "",
        ]

        # ── T2a / T2b ────────────────────────────────────────────────────
        entities = extract_entities_from_report(gt_report)
        entity_a = entities[0] if entities else "lung"
        entity_b = entities[1] if len(entities) > 1 else "heart"

        instr_t2a = f"Describe the findings related to {entity_a} in this CT scan."
        target_t2a = sentences_containing(gt_report, entity_a)
        gen_t2a = generate(model, feat, instr_t2a, args.max_new_tokens)
        print(f"  T2a ({entity_a}): {gen_t2a[:100]}...")
        md += [
            f"**T2a — Entity: {entity_a}**",
            f'- Instruction: "{instr_t2a}"',
            f"- GENERATED: {gen_t2a}",
            f"- GROUND TRUTH (entity-filtered): {target_t2a}",
            "",
        ]

        instr_t2b = f"Describe the findings related to {entity_b} in this CT scan."
        target_t2b = sentences_containing(gt_report, entity_b)
        gen_t2b = generate(model, feat, instr_t2b, args.max_new_tokens)
        print(f"  T2b ({entity_b}): {gen_t2b[:100]}...")
        md += [
            f"**T2b — Entity: {entity_b}**",
            f'- Instruction: "{instr_t2b}"',
            f"- GENERATED: {gen_t2b}",
            f"- GROUND TRUTH (entity-filtered): {target_t2b}",
            "",
        ]

        # ── T3a / T3b ────────────────────────────────────────────────────
        (instr_t3a, abn_a, expect_a), (instr_t3b, abn_b, expect_b) = build_t3_pairs(label_dict)

        gen_t3a = generate(model, feat, instr_t3a, args.max_new_tokens)
        if expect_a is None:
            correct_a_str = "N/A (no positive label available for this scan)"
        else:
            correct = check_yes_no(gen_t3a, expect_a)
            correct_a_str = "YES" if correct else "NO"
            t3_scored += 1
            t3_correct += int(correct)
        print(f"  T3a ({abn_a}, expect Yes): {gen_t3a[:80]}  correct={correct_a_str}")
        md += [
            f"**T3a — Positive: {abn_a}**",
            f'- Instruction: "{instr_t3a}"',
            f"- GENERATED: {gen_t3a}",
            '- GROUND TRUTH: "Yes."',
            f"- Correct: {correct_a_str}",
            "",
        ]

        gen_t3b = generate(model, feat, instr_t3b, args.max_new_tokens)
        if expect_b is None:
            correct_b_str = "N/A (no negative label available for this scan)"
        else:
            correct = check_yes_no(gen_t3b, expect_b)
            correct_b_str = "YES" if correct else "NO"
            t3_scored += 1
            t3_correct += int(correct)
        print(f"  T3b ({abn_b}, expect No): {gen_t3b[:80]}  correct={correct_b_str}")
        md += [
            f"**T3b — Negative: {abn_b}**",
            f'- Instruction: "{instr_t3b}"',
            f"- GENERATED: {gen_t3b}",
            '- GROUND TRUTH: "No."',
            f"- Correct: {correct_b_str}",
            "",
        ]

        # ── Cross-scan T2 block input (entity="lung", fixed) ───────────────
        gen_lung = generate(model, feat, "Describe the findings related to lung in this CT scan.", args.max_new_tokens)
        t2_lung_outputs[stem] = gen_lung

    # ══════════════════════════════════════════════════════════════════════
    # Cross-scan T2 consistency (entity="lung", all 5 scans)
    # ══════════════════════════════════════════════════════════════════════
    print("\n=== Cross-scan T2 consistency (entity=lung) ===")
    md.append('## Cross-scan T2 consistency (entity="lung", all 5 scans)')
    md.append("")
    md.append("| Scan | Generated output |")
    md.append("|---|---|")
    lung_stems = list(t2_lung_outputs.keys())
    lung_texts = [t2_lung_outputs[s] for s in lung_stems]
    for s in lung_stems:
        md.append(f"| {s} | {t2_lung_outputs[s]} |")

    pair_cosines = []
    n = len(lung_texts)
    for i in range(n):
        for j in range(i + 1, n):
            pair_cosines.append(bow_cosine(lung_texts[i], lung_texts[j]))
    mean_cos = sum(pair_cosines) / len(pair_cosines) if pair_cosines else 0.0
    lung_distinct = count_distinct(lung_texts)
    verdict_t2 = "DISTINCT — T2 cross-scan collapse resolved" if lung_distinct == n else "STILL COLLAPSING"

    print(f"Mean pairwise cosine: {mean_cos:.4f}  distinct={lung_distinct}/{n}  verdict={verdict_t2}")
    md.append("")
    md.append(f"Mean pairwise cosine: {mean_cos:.2f}")
    md.append(f"Verdict: [{verdict_t2}]")
    md.append("")

    # ══════════════════════════════════════════════════════════════════════
    # Summary table
    # ══════════════════════════════════════════════════════════════════════
    t1_distinct = count_distinct(t1_outputs)
    md.append("## Summary table")
    md.append("")
    md.append("| Task | Metric | Result |")
    md.append("|---|---|---|")
    md.append(f"| T1 | Distinct outputs across scans | {t1_distinct}/{len(t1_outputs)} |")
    md.append(f"| T2 | Cross-scan distinct (lung entity) | {lung_distinct}/{n} |")
    md.append(f"| T3 | Binary accuracy | {t3_correct}/{t3_scored} |")
    md.append("")

    print(f"\nT1 distinct: {t1_distinct}/{len(t1_outputs)}")
    print(f"T2 (lung) distinct: {lung_distinct}/{n}")
    print(f"T3 accuracy: {t3_correct}/{t3_scored}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md) + "\n")
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    main()
