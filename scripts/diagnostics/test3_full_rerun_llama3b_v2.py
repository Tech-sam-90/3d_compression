#!/usr/bin/env python3
"""Test 3 full rerun — checkpoint_best.pt, Llama-3.2-3B-Instruct + FiLM,
POST-FIX architecture (job 66527771, completed 2026-07-28, best
val_loss=0.4225) — trained from scratch under the current InterSliceAggregator
(top-K sparse + cosine attention + V-FiLM), unlike the original
/scratch/sadeniji/ictc_checkpoints_llama3b checkpoint which predated those
fixes and could not be loaded (see docs/DIAGNOSTIC_REPORT.md).

Read-only diagnostic. Runs:
  Part A — instruction sensitivity: same scan, 5 instructions
  Part B — scan sensitivity: same instruction, 10 scans (PRIMARY TEST)

Same SCAN_INDICES / INSTRUCTIONS_5 / FIXED_INSTRUCTION as every other Test 3
rerun this session so results are directly comparable across checkpoints.
"""
import logging
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
CSV_PATH = "/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv"
CKPT_PATH = "/scratch/sadeniji/ictc_checkpoints_llama3b_v2/checkpoint_best.pt"
CFG_PATH = "configs/ctclip_stage2_llama3b.yaml"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512
MAX_NEW_TOKENS = 200

INSTRUCTIONS_5 = [
    "Generate a radiology report for this CT scan.",
    "Describe any pulmonary findings in this chest CT.",
    "Are there any cardiovascular abnormalities visible?",
    "Summarize mediastinal structures and lymph nodes.",
    "Report any musculoskeletal or pleural findings.",
]
FIXED_INSTRUCTION = "Generate a radiology report for this CT scan."
SCAN_INDICES = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]


def load_scan_list():
    df = pd.read_csv(CSV_PATH)
    rows = []
    for _, row in df.iterrows():
        stem = Path(str(row["VolumeName"])).stem
        if stem.endswith(".nii"):
            stem = stem[:-4]
        pt_path = FEATURES_DIR / f"{stem}.pt"
        if pt_path.exists():
            rows.append((stem, pt_path))
    return rows


def load_feat(pt_path):
    return torch.load(pt_path, weights_only=True).float().reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)


def cosine_stats(vectors):
    v = F.normalize(vectors, dim=-1)
    sim = v @ v.T
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vals = sim[iu[0], iu[1]]
    return {"mean": vals.mean().item(), "min": vals.min().item(), "max": vals.max().item()}, sim


def edit_distance(a: str, b: str) -> int:
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


@torch.no_grad()
def generate_report(model, features_1, instruction: str) -> str:
    device = features_1.device
    etext = model.instruction_encoder([instruction]).float()
    visual = model.projector(features_1, etext)
    visual = model.visual_proj(visual).float()

    inst_enc = model.tokenizer([instruction], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    embed_fn = model.llm.get_input_embeddings()
    inst_embeds = embed_fn(inst_enc.input_ids).float()

    inputs_embeds = torch.cat([visual, inst_embeds], dim=1)
    B, L, _ = inputs_embeds.shape
    attention_mask = torch.ones(B, L, dtype=torch.long, device=device)

    gen_ids = model.llm.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=model.tokenizer.eos_token_id,
        repetition_penalty=1.3,
        no_repeat_ngram_size=3,
    )
    text = model.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text.strip()


def main():
    with open(CFG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    logger.info("Building CTCLIPStage2VLM (conditioning=%s, llm=%s, instruction_encoder=%s) on %s...",
                cfg.get("aggregator_conditioning", "film"), cfg.get("llm_model_name"),
                cfg.get("instruction_encoder_model"), DEVICE)
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
        device=DEVICE,
    )

    logger.info("Loading checkpoint: %s", CKPT_PATH)
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    logger.info("Checkpoint loaded: step=%s epoch=%s val_loss=%.4f",
                ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss", float("nan")))
    model.eval()

    scans = load_scan_list()
    logger.info("Validation set: %d scans with matching features.", len(scans))
    max_idx_needed = max(SCAN_INDICES)
    assert len(scans) > max_idx_needed, f"Need index {max_idx_needed}, only {len(scans)} scans available"

    # ══════════════════════════════════════════════════════════════════════
    # PART A — instruction sensitivity (same scan, 5 instructions)
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("PART A — same scan (index 0), 5 instructions")
    logger.info("=" * 80)
    stem_a, path_a = scans[0]
    feat_a = load_feat(path_a).unsqueeze(0).to(DEVICE)  # (1, D, K, C)
    logger.info("Scan: %s", stem_a)

    with torch.no_grad():
        etext5 = model.instruction_encoder(INSTRUCTIONS_5).float()
        tokens_a = model.projector(feat_a.expand(5, -1, -1, -1), etext5)  # (5, M, embed_dim)
    pooled_a = tokens_a.mean(dim=1)  # (5, embed_dim)
    cos_a, sim_a = cosine_stats(pooled_a)
    logger.info("Part A M-token pooled cosine: mean=%.6f min=%.6f max=%.6f", cos_a["mean"], cos_a["min"], cos_a["max"])
    logger.info("Part A pairwise matrix:\n%s", sim_a)

    reports_a = []
    for i, instr in enumerate(INSTRUCTIONS_5):
        text = generate_report(model, feat_a, instr)
        reports_a.append(text)
        logger.info("[Part A #%d] instruction=%r", i + 1, instr)
        logger.info("[Part A #%d] report: %s", i + 1, text)

    # ══════════════════════════════════════════════════════════════════════
    # PART B — scan sensitivity (same instruction, 10 scans) — PRIMARY TEST
    # ══════════════════════════════════════════════════════════════════════
    logger.info("=" * 80)
    logger.info("PART B — same instruction, 10 scans (indices %s)", SCAN_INDICES)
    logger.info("=" * 80)
    feats_b = []
    stems_b = []
    for idx in SCAN_INDICES:
        stem, path = scans[idx]
        feats_b.append(load_feat(path))
        stems_b.append(stem)
    feats_b = torch.stack(feats_b).to(DEVICE)  # (10, D, K, C)
    logger.info("Scans: %s", stems_b)

    with torch.no_grad():
        etext_fixed = model.instruction_encoder([FIXED_INSTRUCTION] * 10).float()
        tokens_b = model.projector(feats_b, etext_fixed)  # (10, M, embed_dim)
    pooled_b = tokens_b.mean(dim=1)  # (10, embed_dim)
    cos_b, sim_b = cosine_stats(pooled_b)
    logger.info("Part B M-token pooled cosine: mean=%.6f min=%.6f max=%.6f", cos_b["mean"], cos_b["min"], cos_b["max"])
    logger.info("Part B pairwise matrix:\n%s", sim_b)

    reports_b = []
    for i, idx in enumerate(SCAN_INDICES):
        text = generate_report(model, feats_b[i:i + 1], FIXED_INSTRUCTION)
        reports_b.append(text)
        logger.info("[Part B #%d] scan=%s (idx=%d)", i + 1, stems_b[i], idx)
        logger.info("[Part B #%d] report: %s", i + 1, text)

    # ── Text-level collapse diagnostics (Part B) ────────────────────────────
    n = len(reports_b)
    edit_matrix = [[0.0] * n for _ in range(n)]
    exact_dup_pairs = []
    near_dup_pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = edit_distance(reports_b[i], reports_b[j])
            max_len = max(len(reports_b[i]), len(reports_b[j]), 1)
            norm_d = d / max_len
            edit_matrix[i][j] = norm_d

    total_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_pairs += 1
            norm_d = edit_matrix[i][j]
            if reports_b[i] == reports_b[j]:
                exact_dup_pairs.append((i + 1, j + 1))
            if norm_d < 0.05:
                near_dup_pairs.append((i + 1, j + 1))

    collapse_rate = len(near_dup_pairs) / total_pairs

    all_words = []
    for r in reports_b:
        all_words.extend(re.findall(r"\w+", r.lower()))
    unique_words = set(all_words)
    ttr = len(unique_words) / max(len(all_words), 1)

    logger.info("=" * 80)
    logger.info("PART B TEXT-LEVEL COLLAPSE DIAGNOSTIC")
    logger.info("=" * 80)
    logger.info("Edit distance matrix (normalized, 0=identical, 1=completely different):")
    header = "      " + "".join(f"{j+1:>7}" for j in range(n))
    logger.info(header)
    for i in range(n):
        row = "".join(f"{edit_matrix[i][j]:7.3f}" for j in range(n))
        logger.info(f"  {i+1:>3} {row}")
    logger.info("Exact duplicate pairs (%d/%d): %s", len(exact_dup_pairs), total_pairs, exact_dup_pairs)
    logger.info("Near-duplicate pairs, edit_dist<0.05 (%d/%d, superset of exact): %s", len(near_dup_pairs), total_pairs, near_dup_pairs)
    logger.info("COLLAPSE RATE: %.1f%% (%d/%d pairs near-or-exact-duplicate)", collapse_rate * 100, len(near_dup_pairs), total_pairs)
    logger.info("Vocabulary diversity: %d unique words / %d total words, TTR=%.4f", len(unique_words), len(all_words), ttr)

    verdict_band = "0-10% residual noise" if collapse_rate < 0.10 else (
        "10-30% partial collapse" if collapse_rate < 0.30 else "30%+ structural collapse")
    logger.info("Verdict band: %s", verdict_band)

    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Part A cosine: mean=%.6f min=%.6f max=%.6f  (BioMedLM+FiLM=0.9928 [static]; "
                 "BioMedLM+attention-cond=0.8390)", cos_a["mean"], cos_a["min"], cos_a["max"])
    logger.info("Part B cosine: mean=%.6f min=%.6f max=%.6f  (target: meaningfully < 0.90; "
                 "BioMedLM+attention-cond=0.9933)", cos_b["mean"], cos_b["min"], cos_b["max"])
    logger.info("Part B collapse rate: %.1f%% -> %s  (BioMedLM+attention-cond=13.3%%)", collapse_rate * 100, verdict_band)


if __name__ == "__main__":
    main()
