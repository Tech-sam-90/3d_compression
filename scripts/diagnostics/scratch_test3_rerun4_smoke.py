#!/usr/bin/env python3
"""Test 3 rerun 3 — post V-FiLM fix.

Exact same procedure as prior Test 3 reruns, against the Stage 1 checkpoint
loaded into the NEW V-FiLM InterSliceAggregator (cosine attention + V-FiLM).
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
STAGE1_CKPT = "/scratch/sadeniji/smoke_checkpoints_stage2/checkpoint_best.pt"
CFG_PATH = "configs/ctclip_stage1.yaml"
DEVICE = "cpu"

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512

ALLOWED_MISSING_SUFFIXES = [
    "attn_temperature",
    "gamma_v_proj.weight", "gamma_v_proj.bias",
    "beta_v_proj.weight", "beta_v_proj.bias",
]


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


def compute_topk_overlap_and_vfilm(agg, feats_2, etext_2):
    """Replicate forward()'s internals up to topk_idx + V-FiLM gamma/beta,
    for 2 instructions on the same scan."""
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
        Vp = F.linear(kv, Wv, bv)

        gamma_v = 1.0 + agg.gamma_v_proj(etext_2)
        beta_v = agg.beta_v_proj(etext_2)
        Vp = gamma_v.unsqueeze(1) * Vp + beta_v.unsqueeze(1)

        Qh = Qp.view(B, agg._num_tokens, agg._num_heads, head_dim).transpose(1, 2)
        Kh = Kp.view(B, N, agg._num_heads, head_dim).transpose(1, 2)

        Qh = F.normalize(Qh + 1e-8, dim=-1)
        Kh = F.normalize(Kh, dim=-1)
        tau = agg.attn_temperature.clamp(min=1e-4)
        scores = (Qh @ Kh.transpose(-2, -1)) / tau

        effective_top_k = min(agg.top_k, N)
        topk_idx = scores.topk(effective_top_k, dim=-1).indices

        overlaps = []
        for h in range(agg._num_heads):
            for m in range(0, agg._num_tokens, 8):
                s0 = set(topk_idx[0, h, m].tolist())
                s1 = set(topk_idx[1, h, m].tolist())
                overlaps.append(len(s0 & s1) / effective_top_k)
        import statistics
        return statistics.mean(overlaps), scores, gamma_v, beta_v


def main():
    with open(CFG_PATH) as fh:
        cfg = yaml.safe_load(fh)

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
        llm_model_name=cfg.get("llm_model_name"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model"),
        num_labels=cfg.get("num_labels"),
        device=DEVICE,
    )
    ckpt = torch.load(STAGE1_CKPT, map_location=DEVICE)
    result = model.projector.load_state_dict(ckpt["projector"], strict=False)
    real_missing = [k for k in result.missing_keys if not any(k.endswith(s) for s in ALLOWED_MISSING_SUFFIXES)]
    assert not real_missing, f"Unexpected missing keys: {real_missing}"
    assert not result.unexpected_keys, f"Unexpected keys: {result.unexpected_keys}"
    logger.info("projector.load_state_dict strict=False OK: missing=%s (all expected) unexpected=[]",
                result.missing_keys)
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if model.cls_head is not None and "cls_head" in ckpt:
        model.cls_head.load_state_dict(ckpt["cls_head"])
    model.eval()
    logger.info("Stage 1 checkpoint loaded (step=%s, epoch=%s, val_loss=%.4f)",
                ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss"))

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
        visualA = model.projector(scan0.expand(5, -1, -1, -1), etext5)
    visualA_pooled = visualA.mean(dim=1)
    cosA, simA = cosine_stats(visualA_pooled)
    logger.info("PART A (same scan, 5 instructions) cosine: %s", cosA)
    logger.info("PART A matrix:\n%s", simA)

    # Top-K overlap + V-FiLM debug between instructions 1 and 2
    with torch.no_grad():
        etext2 = model.instruction_encoder(instructions_5[:2]).float()
        overlap_pct, scores2, gamma_v2, beta_v2 = compute_topk_overlap_and_vfilm(
            model.projector.stage2, scan0.expand(2, -1, -1, -1), etext2
        )
    logger.info("Top-K overlap between instruction 1 and 2: %.2f%%", overlap_pct * 100)
    logger.info("Score stats: std=%.4f min=%.1f max=%.1f", scores2.std().item(), scores2.min().item(), scores2.max().item())
    logger.info("[DEBUG] gamma_v instr1: %.4f instr2: %.4f", gamma_v2[0].mean().item(), gamma_v2[1].mean().item())
    logger.info("[DEBUG] beta_v  instr1: %.4f  instr2: %.4f", beta_v2[0].mean().item(), beta_v2[1].mean().item())

    # Part B: same instruction, 10 different scans
    with torch.no_grad():
        etext_fixed = model.instruction_encoder([instructions_5[0]] * 10).float()
        visualB = model.projector(feats, etext_fixed)
    visualB_pooled = visualB.mean(dim=1)
    cosB, simB = cosine_stats(visualB_pooled)
    logger.info("PART B (same instruction, 10 scans) cosine: %s", cosB)
    logger.info("PART B matrix:\n%s", simB)

    logger.info("=== GATE CHECK ===")
    cos = cosA["mean"]
    logger.info("Part A mean cosine = %.6f | top-K overlap = %.2f%% | score std = %.4f",
                cos, overlap_pct * 100, scores2.std().item())
    logger.info("History: pre-fix=0.99999988, top-K-only=0.99999988, cosine-attn=0.992790")

    if cos < 0.95:
        logger.info("PRIMARY GATE PASSED: cosine < 0.95. Proceeding to smoke test.")
    else:
        residual_before = 1 - 0.992790
        residual_now = 1 - cos
        reduction_factor = residual_before / residual_now if residual_now > 0 else float("inf")
        logger.warning("PRIMARY GATE FAILED (cosine >= 0.95). Checking secondary gate...")
        logger.info("CT-CLIP floor context: raw feature cosine ~0.87 (Test 1)")
        logger.info("Residual reduction vs previous fix (cosine-attn): %.2fx (need >=100x, i.e. cosine<0.9999)", reduction_factor)
        in_floor_band = 0.87 <= cos < 0.95
        overlap_ok = overlap_pct < 0.80
        std_ok = scores2.std().item() < 10.0
        residual_ok = cos < 0.9999
        logger.info("Secondary gate conditions: in_floor_band[0.87,0.95)=%s overlap<80%%=%s std<10=%s cosine<0.9999=%s",
                     in_floor_band, overlap_ok, std_ok, residual_ok)
        if in_floor_band and overlap_ok and std_ok and residual_ok:
            logger.info("SECONDARY GATE PASSED: treating as PASS per documented reasoning. Proceeding to smoke test.")
        else:
            logger.error("SECONDARY GATE ALSO FAILED. STOP — do not proceed to training.")


if __name__ == "__main__":
    main()
