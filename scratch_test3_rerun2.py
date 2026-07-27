#!/usr/bin/env python3
"""Test 3 rerun 2 — post cosine attention fix.

Exact same procedure as the original Test 3 / Test 3 Rerun, against the
Stage 1 checkpoint loaded into the NEW cosine-attention InterSliceAggregator.
Also computes top-K position overlap between instructions 1 and 2 (new
requirement for this rerun).
"""
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
CSV_PATH = "/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv"
STAGE1_CKPT = "/scratch/sadeniji/ictc_checkpoints_stage1/checkpoint_best.pt"
CFG_PATH = "configs/ctclip_stage1.yaml"
DEVICE = "cpu"

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512


def cosine_stats(vectors):
    v = F.normalize(vectors, dim=-1)
    sim = v @ v.T
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vals = sim[iu[0], iu[1]]
    return {"mean": vals.mean().item(), "std": vals.std().item(),
            "min": vals.min().item(), "max": vals.max().item()}, sim


def load_scans(n):
    df = pd.read_csv(CSV_PATH)
    rows = []
    for _, row in df.iterrows():
        stem = Path(str(row["VolumeName"])).stem
        if stem.endswith(".nii"):
            stem = stem[:-4]
        pt_path = FEATURES_DIR / f"{stem}.pt"
        if pt_path.exists():
            rows.append((stem, pt_path))
        if len(rows) >= n:
            break
    feats, stems = [], []
    for stem, pt_path in rows:
        feat = torch.load(pt_path, weights_only=True).float().reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)
        feats.append(feat)
        stems.append(stem)
    return stems, torch.stack(feats)


def compute_topk_overlap(agg, feats_2, etext_2):
    """Replicate InterSliceAggregator.forward()'s internals up to topk_idx,
    for 2 instructions on the same scan, to measure index-set overlap."""
    with torch.no_grad():
        B, D, K, C = feats_2.shape
        depth_pe = agg.depth_pos_enc(D)
        x = feats_2 + depth_pe.unsqueeze(0).unsqueeze(2)
        kv = x.reshape(B, D * K, C)
        q = agg.depth_queries.unsqueeze(0).expand(B, -1, -1)
        q = agg.film(q, etext_2)
        kv = agg.norm_kv(kv)

        N = kv.shape[1]
        head_dim = agg._embed_dim // agg._num_heads
        Wq, Wk, Wv = agg.cross_attn.in_proj_weight.chunk(3, dim=0)
        bq, bk, bv = agg.cross_attn.in_proj_bias.chunk(3, dim=0)
        Qp = F.linear(q, Wq, bq)
        Kp = F.linear(kv, Wk, bk)
        Qh = Qp.view(B, agg._num_tokens, agg._num_heads, head_dim).transpose(1, 2)
        Kh = Kp.view(B, N, agg._num_heads, head_dim).transpose(1, 2)

        Qh = F.normalize(Qh + 1e-8, dim=-1)
        Kh = F.normalize(Kh, dim=-1)
        tau = agg.attn_temperature.clamp(min=1e-4)
        scores = (Qh @ Kh.transpose(-2, -1)) / tau

        effective_top_k = min(agg.top_k, N)
        topk_idx = scores.topk(effective_top_k, dim=-1).indices  # (2,H,M,k)

        overlaps = []
        for h in range(agg._num_heads):
            for m in range(0, agg._num_tokens, 8):
                s0 = set(topk_idx[0, h, m].tolist())
                s1 = set(topk_idx[1, h, m].tolist())
                overlaps.append(len(s0 & s1) / effective_top_k)
        import statistics
        return statistics.mean(overlaps), scores


def main():
    with open(CFG_PATH) as fh:
        cfg = yaml.safe_load(fh)

    from aadp.models.ctclip_vlm import CTCLIPStage2VLM
    logger.info("aggregator_top_k from config: %s", cfg.get("aggregator_top_k"))
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
        llm_model_name=cfg.get("llm_model_name"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model"),
        num_labels=cfg.get("num_labels"),
        device=DEVICE,
    )
    ckpt = torch.load(STAGE1_CKPT, map_location=DEVICE)
    # attn_temperature is a genuinely new parameter, absent from this
    # checkpoint (saved before this fix) — strict=True would raise a
    # "missing key" error here. strict=False is correct, not a fallback:
    # the freshly-initialized value (0.07) is used since none was saved.
    result = model.projector.load_state_dict(ckpt["projector"], strict=False)
    logger.info("projector.load_state_dict strict=False: missing=%s unexpected=%s",
                result.missing_keys, result.unexpected_keys)
    assert result.missing_keys == ["stage2.attn_temperature"], (
        f"Expected only attn_temperature missing, got {result.missing_keys}"
    )
    assert result.unexpected_keys == [], f"Unexpected keys: {result.unexpected_keys}"
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if model.cls_head is not None and "cls_head" in ckpt:
        model.cls_head.load_state_dict(ckpt["cls_head"])
    model.eval()
    logger.info("Stage 1 checkpoint loaded (step=%s, epoch=%s, val_loss=%.4f)",
                ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss"))
    logger.info("InterSliceAggregator.top_k = %s, attn_temperature = %.4f",
                model.projector.stage2.top_k, model.projector.stage2.attn_temperature.item())

    stems, feats = load_scans(10)
    logger.info("Loaded %d scans: %s", len(stems), stems)

    instructions_5 = [
        "Generate a complete radiology report for this CT scan.",
        "Describe any findings in the bones and spine.",
        "Are there any vascular abnormalities? Describe them.",
        "Compare the left and right lung findings.",
        "Is this CT scan normal? Explain your reasoning.",
    ]

    # Part A: same scan, 5 instructions
    scan0 = feats[0:1]
    with torch.no_grad():
        etext5 = model.instruction_encoder(instructions_5).float()
        q_orig = model.projector.stage2.depth_queries.unsqueeze(0).expand(5, -1, -1)
        q_film = model.projector.stage2.film(q_orig, etext5)
        logger.info("Q L2 norm BEFORE FiLM (same for all, no instr dependence): %.6f",
                    q_orig[0].norm().item())
        logger.info("Q L2 norm AFTER FiLM, per instruction: %s",
                    [round(x, 4) for x in q_film.norm(dim=(1, 2)).tolist()])

        visualA = model.projector(scan0.expand(5, -1, -1, -1), etext5)
    visualA_pooled = visualA.mean(dim=1)
    cosA, simA = cosine_stats(visualA_pooled)
    logger.info("PART A (same scan, 5 instructions) cosine: %s", cosA)
    logger.info("PART A matrix:\n%s", simA)

    # Top-K overlap between instructions 1 and 2 (this rerun's new requirement)
    with torch.no_grad():
        etext2 = model.instruction_encoder(instructions_5[:2]).float()
        overlap_pct, scores2 = compute_topk_overlap(
            model.projector.stage2, scan0.expand(2, -1, -1, -1), etext2
        )
    logger.info("Top-K overlap between instruction 1 and 2: %.2f%%", overlap_pct * 100)
    logger.info("Cosine-attention score stats: std=%.4f min=%.1f max=%.1f mean=%.4f",
                scores2.std().item(), scores2.min().item(), scores2.max().item(), scores2.mean().item())

    # Part B: same instruction, 10 different scans
    with torch.no_grad():
        etext_fixed = model.instruction_encoder([instructions_5[0]] * 10).float()
        visualB = model.projector(feats, etext_fixed)
    visualB_pooled = visualB.mean(dim=1)
    cosB, simB = cosine_stats(visualB_pooled)
    logger.info("PART B (same instruction, 10 scans) cosine: %s", cosB)
    logger.info("PART B matrix:\n%s", simB)

    logger.info("=== GATE CHECK ===")
    logger.info("Part A mean cosine = %.6f (target < 0.90; before this fix = 0.99999988)", cosA["mean"])
    logger.info("Top-K overlap (instr 1 vs 2) = %.2f%% (target < 80%%; before this fix ~93-99%%)", overlap_pct * 100)
    gate_failed = cosA["mean"] >= 0.95 or overlap_pct >= 0.90
    if gate_failed:
        logger.error("GATE FAILED: cosine >= 0.95 OR overlap >= 90%%. STOP — do not proceed to training.")
    elif cosA["mean"] >= 0.90 or overlap_pct >= 0.80:
        logger.warning("GATE BORDERLINE: above target but below hard-stop.")
    else:
        logger.info("GATE PASSED. Proceeding to smoke test.")


if __name__ == "__main__":
    main()
