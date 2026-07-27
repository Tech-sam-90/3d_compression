#!/usr/bin/env python3
"""Component-level diagnostic, Test 6 — LLM visual token sensitivity.

Requires GPU (BioMedLM 2.7B generation). Run via SLURM, not the login node.

For 5 CT scans, generates under 4 conditions using a fixed instruction:
  (a) real M=64 visual tokens for that scan
  (b) real M=64 visual tokens SWAPPED from a different scan
  (c) random Gaussian noise, same shape/dtype as (a)
  (d) all-zeros, same shape/dtype as (a)

Reports character-level edit distance between (a) and each of (b)/(c)/(d).
If (a)-(b)/(a)-(c)/(a)-(d) are all small (near 0), the LLM is ignoring the
visual tokens entirely and generation is instruction-only / boilerplate.
"""
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

FEATURES_DIR = Path("/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
CSV_PATH = "/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv"
N_SCANS = 5
FIXED_INSTRUCTION = "Generate a complete radiology report for this CT scan."
OUT_JSON_TEMPLATE = "/scratch/sadeniji/diagnostic_results/diag_results_test6_{label}.json"

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


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


def run(checkpoint_path: str, label: str, cfg_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s | checkpoint: %s | config: %s", device, checkpoint_path, cfg_path)

    with open(cfg_path) as fh:
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
        device=device,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    model.eval()
    logger.info("Checkpoint loaded (step=%s, epoch=%s, val_loss=%s)",
                ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss"))

    stems, feats = load_scans(N_SCANS)
    feats = feats.to(device)
    logger.info("Loaded %d scans: %s", len(stems), stems)

    M = model.num_tokens
    llm_hidden = model.llm.config.hidden_size

    @torch.no_grad()
    def get_visual(feat_single):
        etext = model.instruction_encoder([FIXED_INSTRUCTION]).float()
        v = model.projector(feat_single.unsqueeze(0), etext)  # (1, M, embed_dim)
        v = model.visual_proj(v).float()                       # (1, M, llm_hidden)
        return v

    @torch.no_grad()
    def generate_from_visual(visual):
        inst_enc = model.tokenizer(
            [FIXED_INSTRUCTION], return_tensors="pt", padding=True, truncation=True, max_length=128,
        ).to(device)
        embed_fn = model.llm.get_input_embeddings()
        inst_embeds = embed_fn(inst_enc.input_ids).float()
        inputs_embeds = torch.cat([visual, inst_embeds], dim=1)
        L = inputs_embeds.shape[1]
        attention_mask = torch.ones(1, L, dtype=torch.long, device=device)
        gen_ids = model.llm.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=200, do_sample=False, pad_token_id=model.tokenizer.eos_token_id,
            repetition_penalty=1.3, no_repeat_ngram_size=3,
        )
        return model.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

    # Precompute real visual tokens for all scans
    real_visuals = [get_visual(feats[i]) for i in range(len(stems))]

    results = {"checkpoint": checkpoint_path, "label": label, "instruction": FIXED_INSTRUCTION, "per_scan": []}

    for i, stem in enumerate(stems):
        logger.info("--- Scan %d/%d: %s ---", i + 1, len(stems), stem)
        v_real = real_visuals[i]
        j = (i + 1) % len(stems)  # swap partner = next scan (wraps)
        v_swapped = real_visuals[j]
        torch.manual_seed(0)
        v_noise = torch.randn(1, M, llm_hidden, device=device, dtype=v_real.dtype)
        v_zeros = torch.zeros(1, M, llm_hidden, device=device, dtype=v_real.dtype)

        out_a = generate_from_visual(v_real)
        out_b = generate_from_visual(v_swapped)
        out_c = generate_from_visual(v_noise)
        out_d = generate_from_visual(v_zeros)

        entry = {
            "scan": stem,
            "swap_partner": stems[j],
            "outputs": {"a_real": out_a, "b_swapped": out_b, "c_noise": out_c, "d_zeros": out_d},
            "edit_distance": {
                "a_vs_b": levenshtein(out_a, out_b),
                "a_vs_c": levenshtein(out_a, out_c),
                "a_vs_d": levenshtein(out_a, out_d),
            },
            "len_a": len(out_a),
        }
        logger.info("  edit distances: a-b=%d a-c=%d a-d=%d (len_a=%d)",
                    entry["edit_distance"]["a_vs_b"], entry["edit_distance"]["a_vs_c"],
                    entry["edit_distance"]["a_vs_d"], entry["len_a"])
        results["per_scan"].append(entry)

    import statistics
    for pair in ("a_vs_b", "a_vs_c", "a_vs_d"):
        vals = [e["edit_distance"][pair] for e in results["per_scan"]]
        results[f"mean_{pair}"] = statistics.mean(vals)

    out_path = OUT_JSON_TEMPLATE.format(label=label)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    logger.info("Wrote results to %s", out_path)
    logger.info("MEAN edit distances: a-b=%.1f a-c=%.1f a-d=%.1f",
                results["mean_a_vs_b"], results["mean_a_vs_c"], results["mean_a_vs_d"])


if __name__ == "__main__":
    import sys
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "/scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt"
    label = sys.argv[2] if len(sys.argv) > 2 else "joint_baseline"
    cfg_path = sys.argv[3] if len(sys.argv) > 3 else "configs/ctclip_stage2.yaml"
    run(ckpt_path, label, cfg_path)
