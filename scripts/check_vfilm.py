#!/usr/bin/env python3
"""check_vfilm.py — diagnostic: has V-FiLM (gamma_v_proj/beta_v_proj) in
InterSliceAggregator (aadp/models/projector/stage2.py) actually learned
anything, or is it still effectively the identity operation it's
zero-initialized to (gamma_v=1, beta_v=0)?

Diagnostic only: no training, no checkpoint writes, no modification to
stage2.py. CHECK 3/4 register forward hooks on stage2.norm_kv (to capture
the exact `kv` tensor used in the real forward pass) and on
stage2.gamma_v_proj/beta_v_proj (to capture their real outputs), then
replicate stage2.py's own (unmodified) Vp/gamma_v/beta_v formula on those
captured values. This makes V_before/V_after byte-identical to what happens
inside a real forward() call without needing to edit forward() itself.

Only the "film" conditioning (InterSliceAggregator) has V-FiLM — the
attention-conditioned ablation (aadp/ablations/attention_conditioned_ctclip_stage2.py)
has no gamma_v_proj/beta_v_proj equivalent (see docs/DIAGNOSTIC3_REPORT.md).

Usage:
    python scripts/check_vfilm.py \\
        --ckpt /scratch/sadeniji/ictc_checkpoints_llama3b_v2/checkpoint_best.pt \\
        --config configs/ctclip_stage2_llama3b_v2.yaml \\
        --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid
"""
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

INSTRUCTION_1 = "Generate a radiology report for this CT scan."
INSTRUCTION_2 = "Describe cardiac and mediastinal structures only."
# Chosen per task spec: most distinctive pathology (cardiomegaly, aortic
# dilatation) among the scans the model was previously shown to miss.
TARGET_STEM = "valid_500_d_1"

_CTCLIP_D, _CTCLIP_K, _CTCLIP_C = 24, 576, 512


def load_feature(feature_dir: Path, stem: str) -> torch.Tensor:
    pt_path = feature_dir / f"{stem}.pt"
    if not pt_path.exists():
        raise FileNotFoundError(f"No feature file for {stem!r} at {pt_path}")
    feat = torch.load(pt_path, weights_only=True).float()
    return feat.reshape(_CTCLIP_D, _CTCLIP_K, _CTCLIP_C)


# ── CHECK 1 — weight norms ───────────────────────────────────────────────

def check1_weight_norms(state_dict: dict) -> dict:
    print("\nCHECK 1 — Weight norms (gamma_v_proj / beta_v_proj)")
    print("-" * 60)
    results = {}
    for key, tensor in state_dict.items():
        if "gamma_v_proj" not in key and "beta_v_proj" not in key:
            continue
        norm = torch.norm(tensor).item()
        max_abs = tensor.abs().max().item()
        if norm < 0.01:
            verdict = "UNTRAINED (near zero init)"
        elif norm > 0.1:
            verdict = "HAS LEARNED something"
        else:
            verdict = "AMBIGUOUS"
        print(f"  {key:40s} norm={norm:.4f}  max_abs={max_abs:.4f}  -> {verdict}")
        results[key] = norm
    return results


# ── CHECK 2 — effective gamma_v range on random input ────────────────────

def check2_gamma_range(state_dict: dict, cond_dim: int, seed: int = 0):
    print("\nCHECK 2 — Effective gamma_v range on random instruction_emb ~ N(0,1)")
    print("-" * 60)
    gamma_w = state_dict["stage2.gamma_v_proj.weight"]
    gamma_b = state_dict["stage2.gamma_v_proj.bias"]
    g = torch.Generator().manual_seed(seed)
    instruction_emb = torch.randn(1, cond_dim, generator=g)
    gamma_v = 1.0 + F.linear(instruction_emb, gamma_w, gamma_b)
    mean, std = gamma_v.mean().item(), gamma_v.std().item()
    lo, hi = gamma_v.min().item(), gamma_v.max().item()
    print(f"  gamma_v: mean={mean:.4f} std={std:.4f} min={lo:.4f} max={hi:.4f}")
    if lo >= 0.99 and hi <= 1.01:
        verdict = "V-FiLM is identity (no effect)"
    elif lo <= 0.9 or hi >= 1.1:
        verdict = "V-FiLM is doing meaningful modulation"
    else:
        verdict = "AMBIGUOUS"
    print(f"  -> {verdict}")
    return lo, hi


# ── CHECK 3 / 4 — real forward pass with capture hooks ───────────────────

def run_with_vfilm_capture(model, feat_1dkc: torch.Tensor, instruction: str):
    """Runs model.projector(features, etext) with hooks capturing the exact
    kv/gamma_raw[/beta] tensors stage2.py's forward() computes internally,
    then replicates its Vp/gamma_v[/beta_v] formula to recover V_before
    (pre-V-FiLM), V_after (post-V-FiLM) — each (B, N, C) — plus gamma_v and
    beta_v themselves (B, C), so callers needing the gamma/beta decomposition
    (e.g. scripts/check_vfilm_scan_diversity.py's beta-dominance check)
    don't need a second, duplicate hook pass.

    Handles BOTH V-FiLM formulas, detected via hasattr(stage2, "beta_v_proj"):
      - Old, unbounded (pre-fix checkpoints, e.g. ictc_checkpoints_llama3b_v2):
        gamma_v = 1.0 + gamma_v_proj(etext); V_after = gamma_v*V_before + beta_v.
        Returns beta_v as a real (B, C) tensor.
      - New, bounded multiplicative-only (post-fix checkpoints, e.g.
        ictc_checkpoints_vfilm_fix — see aadp/models/projector/stage2.py):
        gamma_v = 1.0 + 0.3*tanh(gamma_v_proj(etext)); V_after = gamma_v*V_before.
        beta_v_proj no longer exists at all, so this returns beta_v=None —
        callers must treat that as "no additive term", not a missing value.
    """
    stage2 = model.projector.stage2
    has_beta = hasattr(stage2, "beta_v_proj")
    captured = {}

    def kv_hook(module, inp, out):
        captured["kv"] = out.detach()

    def gamma_hook(module, inp, out):
        captured["gamma_raw"] = out.detach()

    def beta_hook(module, inp, out):
        captured["beta"] = out.detach()

    h1 = stage2.norm_kv.register_forward_hook(kv_hook)
    h2 = stage2.gamma_v_proj.register_forward_hook(gamma_hook)
    h3 = stage2.beta_v_proj.register_forward_hook(beta_hook) if has_beta else None

    with torch.no_grad():
        etext = model.instruction_encoder([instruction]).float()
        _ = model.projector(feat_1dkc, etext)

    h1.remove()
    h2.remove()
    if h3 is not None:
        h3.remove()

    kv = captured["kv"]
    in_proj_weight = stage2.cross_attn.in_proj_weight
    in_proj_bias = stage2.cross_attn.in_proj_bias
    _, _, Wv = in_proj_weight.chunk(3, dim=0)
    _, _, bv = in_proj_bias.chunk(3, dim=0)
    V_before = F.linear(kv, Wv, bv)

    if has_beta:
        gamma_v = 1.0 + captured["gamma_raw"]
        beta_v = captured["beta"]
        V_after = gamma_v.unsqueeze(1) * V_before + beta_v.unsqueeze(1)
    else:
        gamma_v = 1.0 + 0.3 * torch.tanh(captured["gamma_raw"])
        beta_v = None
        V_after = gamma_v.unsqueeze(1) * V_before

    return V_before, V_after, gamma_v, beta_v


def check3_activation_delta(model, feat_1dkc: torch.Tensor):
    print("\nCHECK 3 — Activation delta on a real forward pass")
    print("-" * 60)
    V_before, V_after, _, _ = run_with_vfilm_capture(model, feat_1dkc, INSTRUCTION_1)
    delta = (torch.norm(V_after - V_before) / torch.norm(V_before)).item()
    print(f"  delta = ||V_after - V_before||_F / ||V_before||_F = {delta:.6f}")
    if delta < 0.001:
        verdict = "V-FiLM contributes <0.1% change to values — effectively OFF"
    elif delta > 0.01:
        verdict = "V-FiLM is active"
    else:
        verdict = "AMBIGUOUS"
    print(f"  -> {verdict}")
    return V_before, delta


def check4_cross_instruction_delta(model, feat_1dkc: torch.Tensor, v_before_norm: float):
    print("\nCHECK 4 — Cross-instruction delta (instruction_1 vs instruction_2)")
    print("-" * 60)
    _, V_after_1, _, _ = run_with_vfilm_capture(model, feat_1dkc, INSTRUCTION_1)
    _, V_after_2, _, _ = run_with_vfilm_capture(model, feat_1dkc, INSTRUCTION_2)
    cross_delta = (torch.norm(V_after_1 - V_after_2) / v_before_norm).item()
    print(f"  cross_instr_delta = ||V_after_1 - V_after_2||_F / ||V_before||_F = {cross_delta:.6f}")
    if cross_delta < 0.001:
        verdict = ("V-FiLM produces same values regardless of instruction — "
                    "not contributing to instruction conditioning")
    elif cross_delta > 0.01:
        verdict = "V-FiLM differentiates values by instruction"
    else:
        verdict = "AMBIGUOUS"
    print(f"  -> {verdict}")
    return cross_delta


def build_model(cfg: dict, ckpt: dict, device: str):
    """Construct CTCLIPStage2VLM from a training config and load a
    checkpoint's projector/visual_proj weights. Shared by check_vfilm.py
    and check_vfilm_scan_diversity.py so the two diagnostics build the
    model identically."""
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
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--feature_dir", default="/project/def-uanazodo-ab/sadeniji/ctrate_features/valid")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    conditioning = cfg.get("aggregator_conditioning", "film")
    if conditioning != "film":
        raise SystemExit(
            f"{args.config} has aggregator_conditioning={conditioning!r} — "
            "this diagnostic targets V-FiLM, which only exists in the "
            "'film' InterSliceAggregator, not the attention-conditioned ablation."
        )

    print(f"Loading checkpoint state dict from {args.ckpt} (CPU)...")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    projector_sd = ckpt["projector"]

    # ── CHECK 1 + 2: pure state-dict inspection, no model build needed ──
    weight_norms = check1_weight_norms(projector_sd)
    lo, hi = check2_gamma_range(projector_sd, cond_dim=cfg.get("cond_dim", 2048))

    # ── CHECK 3 + 4: need a real forward pass (instruction encoder + projector) ──
    print(f"\nBuilding CTCLIPStage2VLM on {args.device} for real forward pass "
          "(loads the LLM + instruction encoder — may take a while)...")
    model = build_model(cfg, ckpt, args.device)

    feat = load_feature(Path(args.feature_dir), TARGET_STEM).unsqueeze(0).to(args.device)

    v_before, delta = check3_activation_delta(model, feat)
    v_before_norm = torch.norm(v_before).item()
    cross_delta = check4_cross_instruction_delta(model, feat, v_before_norm)

    # ── Summary ──────────────────────────────────────────────────────────
    gamma_w_norm = weight_norms.get("stage2.gamma_v_proj.weight", 0.0)
    beta_w_norm = weight_norms.get("stage2.beta_v_proj.weight", 0.0)
    gamma_verdict = "UNTRAINED" if gamma_w_norm < 0.01 else ("ACTIVE" if gamma_w_norm > 0.1 else "AMBIGUOUS")
    beta_verdict = "UNTRAINED" if beta_w_norm < 0.01 else ("ACTIVE" if beta_w_norm > 0.1 else "AMBIGUOUS")

    trained = gamma_w_norm > 0.1 and beta_w_norm > 0.1
    untrained = gamma_w_norm < 0.01 and beta_w_norm < 0.01
    active = delta > 0.01
    inactive = delta < 0.001
    differentiating = cross_delta > 0.01

    if untrained and inactive:
        overall = "UNTRAINED AND INACTIVE"
    elif trained and active and differentiating:
        overall = "ACTIVE"
    else:
        overall = "PARTIALLY TRAINED"

    print("\n" + "=" * 60)
    print("V-FiLM Diagnostic")
    print("-" * 60)
    print(f"gamma_v_proj weight norm : {gamma_w_norm:.4f}  -> [{gamma_verdict}]")
    print(f"beta_v_proj  weight norm : {beta_w_norm:.4f}  -> [{beta_verdict}]")
    print(f"gamma range on random input: [{lo:.4f}, {hi:.4f}]")
    print(f"Activation delta (single instr) : {delta:.6f}  -> [{'IDENTITY' if inactive else ('ACTIVE' if active else 'AMBIGUOUS')}]")
    print(f"Cross-instruction delta          : {cross_delta:.6f}  -> [{'NO EFFECT' if cross_delta < 0.001 else ('DIFFERENTIATING' if differentiating else 'AMBIGUOUS')}]")
    print()
    print(f"VERDICT: V-FiLM is [{overall}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
