#!/usr/bin/env python3
"""check_vfilm_scan_diversity.py — does V-FiLM preserve or destroy
cross-scan diversity in the value tensor when the instruction is held
fixed?

check_vfilm.py already established that V-FiLM is ACTIVE (learned nonzero
gamma_v/beta_v, produces a large per-instruction delta). That leaves open
a different failure mode: V-FiLM could be active yet still collapse
DIFFERENT SCANS toward the same output under the SAME instruction — e.g.
if beta_v (a pure function of the instruction, scan-agnostic) dominates
gamma_v * V (the scan-dependent term), V-FiLM would effectively overwrite
scan content with a fixed instruction embedding. This script tests exactly
that, on the collapsed-vs-distinct scan clusters from
docs/DIAGNOSTIC_REPORT.md / docs/DIAGNOSTIC3_REPORT.md.

Diagnostic only: no training, no checkpoint writes, no modification to
stage2.py. Reuses check_vfilm.py's run_with_vfilm_capture() (same
norm_kv/gamma_v_proj/beta_v_proj forward hooks) and build_model() so the
V_before/V_after tensors and model construction are identical to
check_vfilm.py's, not a second, potentially-diverging implementation.

Usage:
    python scripts/check_vfilm_scan_diversity.py \\
        --ckpt /scratch/sadeniji/ictc_checkpoints_llama3b_v2/checkpoint_best.pt \\
        --config configs/ctclip_stage2_llama3b_v2.yaml \\
        --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid
"""
import argparse
from itertools import combinations
from pathlib import Path

import torch
import yaml

from check_vfilm import INSTRUCTION_1 as FIXED_INSTRUCTION
from check_vfilm import build_model, load_feature, run_with_vfilm_capture

COLLAPSED = ["valid_131_a_1", "valid_634_a_1", "valid_1016_b_2"]
DISTINCT = ["valid_1_a_1", "valid_382_c_2", "valid_500_d_1"]
ALL_SCANS = COLLAPSED + DISTINCT


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return (torch.dot(a, b) / (torch.norm(a) * torch.norm(b))).item()


def pair_group(stem_i: str, stem_j: str) -> str:
    if stem_i in COLLAPSED and stem_j in COLLAPSED:
        return "collapsed-collapsed"
    if stem_i in DISTINCT and stem_j in DISTINCT:
        return "distinct-distinct"
    return "cross-group"


def mean_std(values: list) -> tuple:
    t = torch.tensor(values)
    return t.mean().item(), (t.std().item() if len(values) > 1 else 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature_dir", default="/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    conditioning = cfg.get("aggregator_conditioning", "film")
    if conditioning != "film":
        raise SystemExit(
            f"{args.config} has aggregator_conditioning={conditioning!r} — "
            "V-FiLM only exists in the 'film' InterSliceAggregator."
        )

    print(f"Loading checkpoint from {args.ckpt} (CPU)...")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    print(f"Building CTCLIPStage2VLM on {args.device} "
          "(loads the LLM + instruction encoder — may take a while)...")
    model = build_model(cfg, ckpt, args.device)

    feature_dir = Path(args.feature_dir)

    pooled_before, pooled_after = {}, {}
    gamma_norms, beta_norms = {}, {}

    for stem in ALL_SCANS:
        feat = load_feature(feature_dir, stem).unsqueeze(0).to(args.device)
        V_before, V_after, gamma_v, beta_v = run_with_vfilm_capture(model, feat, FIXED_INSTRUCTION)

        pooled_before[stem] = V_before.mean(dim=1).squeeze(0)   # (C,)
        pooled_after[stem] = V_after.mean(dim=1).squeeze(0)     # (C,)

        # Beta dominance decomposition at full token resolution (N tokens),
        # matching how V_after itself is actually computed (broadcast add),
        # not at the pooled/summarized level. beta_v is None when the
        # checkpoint's architecture has no beta_v_proj at all (the bounded
        # V-FiLM fix, aadp/models/projector/stage2.py) — that's not a
        # missing value, it's a real "zero additive term" data point, so
        # beta_norms is correctly 0.0 for every scan in that case rather
        # than being skipped.
        N = V_before.shape[1]
        gamma_term = gamma_v.unsqueeze(1) * V_before                       # (B, N, C)
        gamma_norms[stem] = torch.norm(gamma_term).item()
        if beta_v is not None:
            beta_term = beta_v.unsqueeze(1).expand(-1, N, -1)              # (B, N, C)
            beta_norms[stem] = torch.norm(beta_term).item()
        else:
            beta_norms[stem] = 0.0

        print(f"  {stem:20s} done "
              f"(||gamma*V||={gamma_norms[stem]:.2f}, ||beta||={beta_norms[stem]:.2f})")

    # ── 1. PAIRWISE COSINE ───────────────────────────────────────────────
    groups_before = {"collapsed-collapsed": [], "distinct-distinct": [], "cross-group": []}
    groups_after = {"collapsed-collapsed": [], "distinct-distinct": [], "cross-group": []}

    for stem_i, stem_j in combinations(ALL_SCANS, 2):
        grp = pair_group(stem_i, stem_j)
        groups_before[grp].append(cosine(pooled_before[stem_i], pooled_before[stem_j]))
        groups_after[grp].append(cosine(pooled_after[stem_i], pooled_after[stem_j]))

    # ── 2. DIVERSITY RATIO ────────────────────────────────────────────────
    stack_before = torch.stack([pooled_before[s] for s in ALL_SCANS])   # (6, C)
    stack_after = torch.stack([pooled_after[s] for s in ALL_SCANS])     # (6, C)
    diversity_before = stack_before.std(dim=0).mean().item()
    diversity_after = stack_after.std(dim=0).mean().item()
    ratio = diversity_after / diversity_before

    # ── 3. BETA DOMINANCE ─────────────────────────────────────────────────
    mean_gamma_norm = sum(gamma_norms.values()) / len(gamma_norms)
    mean_beta_norm = sum(beta_norms.values()) / len(beta_norms)
    beta_fraction = mean_beta_norm / (mean_beta_norm + mean_gamma_norm) * 100

    # ── Output ───────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("V-FiLM Scan Diversity Diagnostic")
    print("=" * 70)
    print("Pairwise cosine similarity (same instruction, different scans):\n")
    print(f"  {'Pair type':22s}  {'V_before (mean±std)':22s}  {'V_after (mean±std)'}")
    for grp in ("collapsed-collapsed", "distinct-distinct", "cross-group"):
        mb, sb = mean_std(groups_before[grp])
        ma, sa = mean_std(groups_after[grp])
        print(f"  {grp:22s}  {mb:.3f} ± {sb:.3f}          {ma:.3f} ± {sa:.3f}")

    print(f"\nDiversity ratio (after/before): {ratio:.2f}")
    if ratio < 0.5:
        print("  -> [COLLAPSE: V-FiLM compresses scan diversity]")
    elif ratio > 0.9:
        print("  -> [PRESERVED: scan diversity survives V-FiLM]")
    else:
        print("  -> [PARTIAL: some diversity loss, inspect pairwise cosines above]")

    print("\nBeta dominance (mean across 6 scans):")
    print(f"  ||gamma × V||  = {mean_gamma_norm:.2f}")
    print(f"  ||beta||       = {mean_beta_norm:.2f}")
    print(f"  beta fraction  = {beta_fraction:.1f}%")
    if mean_beta_norm == 0.0:
        print("  -> [N/A: beta_v_proj removed entirely by the V-FiLM fix — "
              "no additive term exists, so beta dominance is structurally "
              "impossible rather than measured-and-low]")
    elif beta_fraction > 50.0:
        print("  -> [BETA DOMINATES: V-FiLM ignores scan content]")
    else:
        print("  -> [BALANCED: both gamma and beta contribute]")

    print("\nVERDICT:")
    if ratio < 0.5:
        print("  [V-FiLM IS collapsing scan diversity — fix: constrain gamma range]")
    elif ratio > 0.9:
        print("  [V-FiLM preserves scan diversity — collapse is downstream in the LLM]")
    else:
        print("  [PARTIAL diversity loss — neither fully collapsing nor fully "
              "preserving; inspect the pairwise cosine matrix and beta fraction above]")
    print("=" * 70)


if __name__ == "__main__":
    main()
