#!/usr/bin/env python3
"""Component-level diagnostics, Tests 1 (adapted) through 5.

Run on the login node (CPU). Loads the Stage 1 checkpoint (aggregator +
FiLM + instruction encoder + cls_head) and 50 CT-RATE *valid*-split scans
(no test split exists — see DIAGNOSTIC_REPORT.md).
"""
import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

DEVICE = "cpu"
N_SCANS = 50
FEATURES_DIR = Path("/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
CSV_PATH = "/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv"
STAGE1_CKPT = "/scratch/sadeniji/ictc_checkpoints_stage1/checkpoint_best.pt"
CFG_PATH = "configs/ctclip_stage1.yaml"
OUT_JSON = "/tmp/claude-3160670/-home-sadeniji-3d-compression/cc7d4520-5528-4f44-a5fb-606c618a823b/scratchpad/diag_results_1to5.json"

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512


def load_scans(n):
    df = pd.read_csv(CSV_PATH)
    label_cols = None
    # reuse same detection logic as CTCLIPFeatureDataset
    from aadp.data.ctclip_feature_dataset import _NON_LABEL_COLS
    meta_cols = _NON_LABEL_COLS | set()
    candidate = [c for c in df.columns if c not in meta_cols and pd.api.types.is_integer_dtype(df[c])]
    for c in df.columns:
        if c not in meta_cols and c not in candidate:
            try:
                uv = df[c].dropna().unique()
                if len(uv) > 0 and set(uv).issubset({0, 1, 0.0, 1.0}):
                    candidate.append(c)
            except Exception:
                pass
    label_cols = sorted(candidate)
    logger.info("Detected %d label columns", len(label_cols))

    rows = []
    for _, row in df.iterrows():
        stem = Path(str(row["VolumeName"])).stem
        if stem.endswith(".nii"):
            stem = stem[:-4]
        pt_path = FEATURES_DIR / f"{stem}.pt"
        if pt_path.exists():
            rows.append((stem, row, pt_path))
        if len(rows) >= n:
            break
    logger.info("Loaded %d / %d requested scans", len(rows), n)

    feats, labels, stems = [], [], []
    for stem, row, pt_path in rows:
        feat = torch.load(pt_path, weights_only=True).float().reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)
        feats.append(feat)
        lab = [int(row.get(c, 0)) if not pd.isna(row.get(c, 0)) else 0 for c in label_cols]
        labels.append(lab)
        stems.append(stem)
    return stems, torch.stack(feats), np.array(labels), label_cols


def cosine_stats(vectors):
    """vectors: (N, D) tensor. Returns dict of pairwise-cosine stats (upper triangle, excl diagonal)."""
    v = F.normalize(vectors, dim=-1)
    sim = v @ v.T
    n = sim.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    vals = sim[iu[0], iu[1]]
    return {
        "mean": vals.mean().item(),
        "std": vals.std().item(),
        "min": vals.min().item(),
        "max": vals.max().item(),
    }, sim


def linear_probe_auroc(X, Y, label_cols, n_train=30, n_test=20, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    train_idx, test_idx = idx[:n_train], idx[n_train:n_train + n_test]
    per_class = {}
    aurocs = []
    for ci, name in enumerate(label_cols):
        y_train = Y[train_idx, ci]
        y_test = Y[test_idx, ci]
        if len(set(y_train.tolist())) < 2 or len(set(y_test.tolist())) < 2:
            per_class[name] = None  # can't compute AUROC, degenerate class
            continue
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y_train)
        prob = clf.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y_test, prob)
        per_class[name] = auc
        aurocs.append(auc)
    mean_auc = float(np.mean(aurocs)) if aurocs else float("nan")
    return mean_auc, per_class, len(aurocs)


def main():
    results = {}

    logger.info("=== Loading %d scans from VALID split ===", N_SCANS)
    stems, feats, labels, label_cols = load_scans(N_SCANS)
    results["n_scans_loaded"] = len(stems)
    results["label_cols"] = label_cols
    results["label_positive_counts"] = {c: int(labels[:, i].sum()) for i, c in enumerate(label_cols)}

    # ── TEST 1 (adapted): raw CT-CLIP feature quality, mean-pooled ──────────
    logger.info("=== TEST 1 (adapted): raw feature cosine sim + linear probe ===")
    raw_pooled = feats.mean(dim=(1, 2))  # (N, 512) mean over D,K
    cos1, _ = cosine_stats(raw_pooled)
    mean_auc1, per_class1, n_valid1 = linear_probe_auroc(raw_pooled.numpy(), labels, label_cols)
    results["test1_adapted"] = {
        "cosine": cos1,
        "mean_auroc": mean_auc1,
        "n_classes_with_valid_auroc": n_valid1,
        "per_class_auroc": per_class1,
    }
    logger.info("Test1 cosine: %s  mean_auroc=%.4f (n_classes=%d)", cos1, mean_auc1, n_valid1)

    # ── Build model components (instruction encoder + projector + cls_head) ──
    logger.info("=== Building model components from Stage 1 config/checkpoint ===")
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
        llm_model_name=cfg.get("llm_model_name"),
        llm_frozen=cfg.get("llm_frozen", False),
        llm_lora=cfg.get("llm_lora"),
        instruction_encoder_model=cfg.get("instruction_encoder_model"),
        num_labels=cfg.get("num_labels"),
        device=DEVICE,
    )
    ckpt = torch.load(STAGE1_CKPT, map_location=DEVICE)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if model.cls_head is not None and "cls_head" in ckpt:
        model.cls_head.load_state_dict(ckpt["cls_head"])
    model.eval()
    logger.info("Stage 1 checkpoint loaded (step=%s, epoch=%s, val_loss=%.4f)",
                ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss"))

    NEUTRAL_INSTR = "Describe the findings of this CT scan."

    # ── TEST 2: Perceiver aggregator, FiLM frozen to identity ───────────────
    logger.info("=== TEST 2: Perceiver output with FiLM forced to identity ===")
    from aadp.models.film import NullFiLMLayer
    real_film = model.projector.stage2.film
    identity_film = NullFiLMLayer(cond_dim=cfg.get("cond_dim", 2048), target_dim=cfg.get("embed_dim", 512))
    model.projector.stage2.film = identity_film
    with torch.no_grad():
        etext_neutral = model.instruction_encoder([NEUTRAL_INSTR] * len(stems)).float()
        visual_identity = model.projector(feats, etext_neutral)  # (N, M, embed_dim)
    model.projector.stage2.film = real_film  # restore

    visual_identity_pooled = visual_identity.mean(dim=1)  # (N, embed_dim)
    cos2, _ = cosine_stats(visual_identity_pooled)
    mean_auc2, per_class2, n_valid2 = linear_probe_auroc(visual_identity_pooled.numpy(), labels, label_cols)
    results["test2_perceiver_identity_film"] = {
        "cosine": cos2,
        "mean_auroc": mean_auc2,
        "n_classes_with_valid_auroc": n_valid2,
        "per_class_auroc": per_class2,
        "comparison_to_test1": {
            "cosine_delta": cos2["mean"] - cos1["mean"],
            "auroc_delta": mean_auc2 - mean_auc1,
        },
    }
    logger.info("Test2 cosine: %s  mean_auroc=%.4f  (delta vs Test1: cos=%.4f auroc=%.4f)",
                cos2, mean_auc2, cos2["mean"] - cos1["mean"], mean_auc2 - mean_auc1)

    # ── TEST 3: FiLM / instruction conditioning (real FiLM active) ──────────
    logger.info("=== TEST 3: FiLM conditioning sensitivity ===")
    instructions_5 = [
        "Generate a complete radiology report for this CT scan.",
        "Describe any findings in the bones and spine.",
        "Are there any vascular abnormalities? Describe them.",
        "Compare the left and right lung findings.",
        "Is this CT scan normal? Explain your reasoning.",
    ]
    # Part A: same scan, 5 instructions
    scan0 = feats[0:1]  # (1, D, K, C)
    with torch.no_grad():
        etext5 = model.instruction_encoder(instructions_5).float()
        visualA = model.projector(scan0.expand(5, -1, -1, -1), etext5)  # (5, M, embed)
    visualA_pooled = visualA.mean(dim=1)
    cosA, simA = cosine_stats(visualA_pooled)

    # Part B: same instruction, 10 different scans
    scans10 = feats[:10]
    with torch.no_grad():
        etext_fixed = model.instruction_encoder([instructions_5[0]] * 10).float()
        visualB = model.projector(scans10, etext_fixed)
    visualB_pooled = visualB.mean(dim=1)
    cosB, simB = cosine_stats(visualB_pooled)

    inverted = cosA["mean"] > cosB["mean"]
    results["test3_film_conditioning"] = {
        "part_a_same_scan_5_instructions": {
            "cosine": cosA,
            "matrix": simA.tolist(),
            "instructions": instructions_5,
        },
        "part_b_same_instruction_10_scans": {
            "cosine": cosB,
            "matrix": simB.tolist(),
            "scan_stems": stems[:10],
        },
        "inverted_conditioning": inverted,
    }
    logger.info("Test3 Part A (instr vary) cosine mean=%.4f | Part B (scan vary) cosine mean=%.4f | inverted=%s",
                cosA["mean"], cosB["mean"], inverted)

    # ── TEST 4: Instruction encoder embedding quality ────────────────────────
    logger.info("=== TEST 4: instruction encoder embedding quality ===")
    set_a = instructions_5 + [
        "Identify any pleural effusion.",
        "What are the mediastinal findings?",
        "Describe the cardiac silhouette.",
    ]
    set_b = [
        "Generate a complete radiology report for this CT scan.",
        "Please write a full radiology report based on this CT.",
        "Provide a detailed radiology report for the CT scan shown.",
        "Create a comprehensive report describing the CT findings.",
        "Write a radiology report covering all findings in this CT.",
    ]
    with torch.no_grad():
        emb_a = model.instruction_encoder(set_a).float()
        emb_b = model.instruction_encoder(set_b).float()
    cos_a, sim_a = cosine_stats(emb_a)
    cos_b, sim_b = cosine_stats(emb_b)
    results["test4_instruction_encoder"] = {
        "set_a_distinct": {"cosine": cos_a, "matrix": sim_a.tolist(), "instructions": set_a},
        "set_b_paraphrases": {"cosine": cos_b, "matrix": sim_b.tolist(), "instructions": set_b},
        "collapsing": abs(cos_a["mean"] - cos_b["mean"]) < 0.05,
    }
    logger.info("Test4 Set A (distinct) mean=%.4f | Set B (paraphrases) mean=%.4f", cos_a["mean"], cos_b["mean"])

    # ── TEST 5: cls_head AUROC ────────────────────────────────────────────────
    logger.info("=== TEST 5: cls_head AUROC (Stage 1 checkpoint, neutral instruction) ===")
    if model.cls_head is None:
        results["test5_cls_head_auroc"] = {"skipped": "model.cls_head is None"}
    else:
        with torch.no_grad():
            etext_all = model.instruction_encoder([NEUTRAL_INSTR] * len(stems)).float()
            visual_all = model.projector(feats, etext_all)
            visual_all_proj = model.visual_proj(visual_all).float()
            pooled = visual_all_proj.mean(dim=1)
            logits = model.cls_head(pooled)
            probs = torch.sigmoid(logits).numpy()

        per_class_auc5 = {}
        aurocs5 = []
        for i, name in enumerate(label_cols):
            y = labels[:, i]
            if len(set(y.tolist())) < 2:
                per_class_auc5[name] = None
                continue
            auc = roc_auc_score(y, probs[:, i])
            per_class_auc5[name] = auc
            aurocs5.append(auc)
        mean_auc5 = float(np.mean(aurocs5)) if aurocs5 else float("nan")
        n_gt_60 = sum(1 for a in aurocs5 if a > 0.60)
        n_gt_70 = sum(1 for a in aurocs5 if a > 0.70)
        results["test5_cls_head_auroc"] = {
            "mean_auroc": mean_auc5,
            "n_classes_evaluated": len(aurocs5),
            "n_classes_auroc_gt_0.60": n_gt_60,
            "n_classes_auroc_gt_0.70": n_gt_70,
            "per_class_auroc": per_class_auc5,
        }
        logger.info("Test5 mean_auroc=%.4f  n>0.60=%d  n>0.70=%d (of %d evaluated / 18 total)",
                    mean_auc5, n_gt_60, n_gt_70, len(aurocs5))

    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    logger.info("Wrote results to %s", OUT_JSON)


if __name__ == "__main__":
    main()
