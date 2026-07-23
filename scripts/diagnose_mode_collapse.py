#!/usr/bin/env python3
"""diagnose_mode_collapse.py — isolate whether the visual conditioning
(InterSliceAggregator) or the instruction conditioning is failing to reach
the LLM, given the trained checkpoint produces near-identical output
regardless of input.

Loads a trained CTCLIPStage2VLM checkpoint and runs:

  1. SAME instruction, 8 DIFFERENT CT scans (val set) — prints all 8
     generated outputs, and the pairwise cosine similarity matrix of the
     M visual tokens InterSliceAggregator produces for each (flattened to
     one vector per sample). If outputs are near-identical AND similarity
     > 0.95, the aggregator is not differentiating CT inputs.

  2. SAME CT scan, 5 DIFFERENT instructions — prints all 5 generated
     outputs. If they're all near-identical, the instruction signal isn't
     varying the output.

  3. ONE training forward+backward pass — logs gradient norms for
     InterSliceAggregator parameters separately from LoRA parameters, to
     check whether the aggregator is receiving learning signal at all.

Reports findings only — does not propose or apply any fix.

Usage:
    python scripts/diagnose_mode_collapse.py \\
        --config configs/ctclip_stage2.yaml \\
        --checkpoint /scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt
"""

import argparse
import itertools
import logging

import torch
import torch.nn.functional as F
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _build_model(cfg, device):
    from aadp.models.ctclip_vlm import CTCLIPStage2VLM

    return CTCLIPStage2VLM(
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


def _load_checkpoint(model, checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    logger.info(
        "Checkpoint loaded (step=%s, epoch=%s, val_loss=%.4f).",
        ckpt.get("step"), ckpt.get("epoch"), ckpt.get("val_loss", float("nan")),
    )


@torch.no_grad()
def _generate(model, features, instruction, device):
    features = features.unsqueeze(0).to(device)
    out = model(features, [instruction], training=False)
    return model.tokenizer.batch_decode(out["generated_ids"], skip_special_tokens=True)[0]


@torch.no_grad()
def _visual_tokens(model, features, instruction, device):
    """Return the flattened (M*embed_dim,) InterSliceAggregator output."""
    features = features.unsqueeze(0).to(device)
    etext = model.instruction_encoder([instruction]).float()
    visual = model.projector(features, etext)  # (1, M, embed_dim)
    return visual.squeeze(0).flatten()          # (M*embed_dim,)


def _cosine_matrix(vectors):
    n = len(vectors)
    mat = torch.zeros(n, n)
    for i, j in itertools.product(range(n), range(n)):
        mat[i, j] = F.cosine_similarity(vectors[i].unsqueeze(0), vectors[j].unsqueeze(0)).item()
    return mat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    device = cfg.get("device", "cuda")
    if not torch.cuda.is_available() and device == "cuda":
        logger.warning("CUDA unavailable — falling back to CPU.")
        device = "cpu"

    from aadp.data.ctclip_feature_dataset import CTCLIPFeatureDataset

    logger.info("Loading validation dataset from %s", cfg["features_valid_dir"])
    val_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_valid_dir"],
        csv_path=cfg["ctrate_csv_valid"],
        tasks=["T1"],
        max_samples=None,
    )
    train_ds = CTCLIPFeatureDataset(
        features_dir=cfg["features_train_dir"],
        csv_path=cfg["ctrate_csv_train"],
        tasks=["T1"],
        max_samples=8,
    )

    model = _build_model(cfg, device)
    _load_checkpoint(model, args.checkpoint, device)
    model.eval()

    FIXED_INSTRUCTION = "Generate a radiology report for this CT scan."

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: same instruction, 8 different CT scans
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 1 — Fixed instruction, 8 different CT scans")
    print(f'Instruction (all 8): "{FIXED_INSTRUCTION}"')
    print("=" * 90)

    n_scans = min(8, len(val_ds))
    scan_indices = list(range(0, len(val_ds), max(1, len(val_ds) // n_scans)))[:n_scans]

    outputs_1 = []
    visual_vecs_1 = []
    patient_ids_1 = []
    for idx in scan_indices:
        item = val_ds[idx]
        features = item["features"]
        patient_id = item["patient_id"]
        text = _generate(model, features, FIXED_INSTRUCTION, device)
        vec = _visual_tokens(model, features, FIXED_INSTRUCTION, device)
        outputs_1.append(text)
        visual_vecs_1.append(vec.cpu())
        patient_ids_1.append(patient_id)
        print(f"\n--- Scan {len(outputs_1) - 1} (patient_id={patient_id}) ---")
        print(text)

    cos_mat_1 = _cosine_matrix(visual_vecs_1)
    print("\n--- Pairwise cosine similarity of InterSliceAggregator output (8x8) ---")
    header = "        " + "  ".join(f"S{i}" for i in range(len(outputs_1)))
    print(header)
    for i in range(len(outputs_1)):
        row = "  ".join(f"{cos_mat_1[i, j].item():.4f}" for j in range(len(outputs_1)))
        print(f"S{i:<6} {row}")
    off_diag = cos_mat_1[~torch.eye(len(outputs_1), dtype=torch.bool)]
    print(f"\nOff-diagonal mean cosine similarity: {off_diag.mean().item():.4f}")
    print(f"Off-diagonal max cosine similarity:  {off_diag.max().item():.4f}")
    print(f"Off-diagonal min cosine similarity:  {off_diag.min().item():.4f}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: same CT scan, 5 different instructions
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 2 — Fixed CT scan, 5 different instructions")
    print("=" * 90)

    fixed_item = val_ds[scan_indices[0]]
    fixed_features = fixed_item["features"]
    print(f"CT scan: patient_id={fixed_item['patient_id']}")

    instructions_2 = [
        "Describe the lungs in this CT scan.",
        "Describe the aorta in this CT scan.",
        "Describe the liver in this CT scan.",
        "Describe any fractures in this CT scan.",
        "Describe the overall findings in this CT scan.",
    ]
    outputs_2 = []
    for instr in instructions_2:
        text = _generate(model, fixed_features, instr, device)
        outputs_2.append(text)
        print(f'\n--- Instruction: "{instr}" ---')
        print(text)

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: gradient norms — InterSliceAggregator vs LoRA
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("TEST 3 — Gradient norms: InterSliceAggregator vs LoRA (one train step)")
    print("=" * 90)

    model.train()
    model.zero_grad()

    train_item = train_ds[0]
    features = train_item["features"].unsqueeze(0).to(device)
    target = train_item["target"]
    target_enc = model.tokenizer(
        [target], return_tensors="pt", padding=True, truncation=True, max_length=256,
    ).input_ids.to(device)

    out = model(features, [FIXED_INSTRUCTION], report_tokens=target_enc, training=True)
    loss = out["loss"]
    loss.backward()

    def _grad_norm(params):
        grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
        if not grads:
            return None, 0
        full = torch.cat(grads)
        return full.norm().item(), len(grads)

    aggregator_params = list(model.projector.stage2.parameters())
    lora_params = [p for n, p in model.llm.named_parameters() if "lora_" in n]
    visual_proj_params = list(model.visual_proj.parameters())

    agg_norm, agg_n = _grad_norm(aggregator_params)
    lora_norm, lora_n = _grad_norm(lora_params)
    vp_norm, vp_n = _grad_norm(visual_proj_params)

    print(f"Training loss on this sample: {loss.item():.6f}")
    print(f"InterSliceAggregator (model.projector.stage2): grad_norm={agg_norm}, "
          f"{agg_n}/{len(aggregator_params)} tensors have gradients")
    print(f"visual_proj:                                   grad_norm={vp_norm}, "
          f"{vp_n}/{len(visual_proj_params)} tensors have gradients")
    print(f"LoRA (llm lora_* params):                       grad_norm={lora_norm}, "
          f"{lora_n}/{len(lora_params)} tensors have gradients")

    print("\n--- Per-submodule gradient norm breakdown (InterSliceAggregator) ---")
    for name, module in model.projector.stage2.named_children():
        params = list(module.parameters())
        norm, n = _grad_norm(params)
        print(f"  {name}: grad_norm={norm}, {n}/{len(params)} tensors have gradients")

    print("\n" + "=" * 90)
    print("DONE. No fixes proposed — findings only.")
    print("=" * 90)


if __name__ == "__main__":
    main()
