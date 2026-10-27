#!/usr/bin/env python3
"""qualitative_eval_v3.py — raw-volume (volume=) evaluation for the
end-to-end (unfrozen ViT) checkpoints from scripts/train_ctclip.py's
_train_e2e() / configs/ctclip_e2e_stage{1,2}.yaml.

Deliberately NOT a features= script: qualitative_eval_v2.py loads
pre-extracted .pt features and hand-rolls its own generation logic
duplicating pieces of CTCLIPStage2VLM.forward(). This script instead uses
CTRawDataset (raw NIfTI volumes) and CTCLIPStage2VLM.generate() — a thin
wrapper this file's own Step 2 added around forward()'s existing
volume=-aware inference path, so there's exactly one generation code path,
not two that can drift apart.

Checkpoint format: supports BOTH
  1. The new DeepSpeed ZeRO-3 flat state dict written by
     model_engine.save_16bit_model() at end of stage1
     (scripts/train_ctclip.py's _train_e2e()) — checked for under
     checkpoint_dir as either checkpoint_latest_consolidated.pt (this
     project's own filename) or pytorch_model.bin (the generic
     save_16bit_model() default, in case a run used that instead).
  2. The OLD piecewise format from the pre-e2e _save_checkpoint()/
     _load_checkpoint() in train_ctclip.py — {"projector":...,
     "visual_proj":...,  optionally "llm_lora":...} — as a single .pt file,
     e.g. checkpoint_best.pt. This is what every checkpoint that predates
     this end-to-end redesign uses (including the smoke-test checkpoint,
     /scratch/sadeniji/ictc_checkpoints_vfilm_fix/checkpoint_best.pt).
A DeepSpeed ZeRO-3 sharded save_checkpoint() directory (e.g. a raw
checkpoint_best/ subdirectory) is NOT directly loadable this way — that's
exactly why save_16bit_model() exists; point --checkpoint-dir at the parent
directory containing the consolidated file, not the sharded tag directory.

T3's ground-truth format is "Yes. <evidence sentence>" / "No." (verified
directly against aadp/data/ctclip_raw_dataset.py — NOT "abnormal"/"normal").

Usage:
    python scripts/qualitative_eval_v3.py \\
        --checkpoint-dir /scratch/sadeniji/ictc_checkpoints_e2e_stage1 \\
        --config configs/ctclip_e2e_stage1.yaml \\
        --num-samples 200 \\
        --output /scratch/sadeniji/ictc/eval/stage1_eval.csv
"""
import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, "/cvmfs/soft.computecanada.ca/easybuild/software/2023/"
                    "x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages")

import torch
import yaml
from torch.utils.data import DataLoader

from aadp.data.ctclip_raw_dataset import CTRawDataset, collate_fn
from aadp.data.instruction_builder import sentences_containing
from aadp.models.ctclip_vlm import CTCLIPStage2VLM

_ENTITY_RE = re.compile(r"related to (.+?) in this CT scan", re.IGNORECASE)


# ── Checkpoint loading ──────────────────────────────────────────────────────

def load_checkpoint(model: CTCLIPStage2VLM, checkpoint_dir: Path) -> str:
    """Tries, in order: the consolidated ZeRO-3 flat state dict (this
    project's filename, then the generic DeepSpeed default), then the old
    piecewise single-file format. Returns a description of what was loaded."""
    consolidated_candidates = [
        checkpoint_dir / "checkpoint_latest_consolidated.pt",
        checkpoint_dir / "pytorch_model.bin",
    ]
    for path in consolidated_candidates:
        if path.exists():
            state = torch.load(path, map_location="cpu")
            result = model.load_state_dict(state, strict=False)
            if result.missing_keys or result.unexpected_keys:
                print(f"WARNING: loading {path} — missing={len(result.missing_keys)} "
                      f"unexpected={len(result.unexpected_keys)} keys")
            return f"consolidated flat state dict: {path}"

    old_format_path = checkpoint_dir / "checkpoint_best.pt"
    if not old_format_path.exists():
        # Also accept checkpoint_dir itself being the .pt file directly.
        if checkpoint_dir.is_file():
            old_format_path = checkpoint_dir
        else:
            raise FileNotFoundError(
                f"No checkpoint found under {checkpoint_dir}. Tried: "
                f"{[str(p) for p in consolidated_candidates]}, {old_format_path}. "
                "If this is a DeepSpeed ZeRO-3 run, point --checkpoint-dir at the "
                "directory containing the consolidated file written by "
                "model_engine.save_16bit_model() (the checkpoint_dir passed to "
                "_train_e2e, not a sharded checkpoint_best/ tag subdirectory)."
            )

    ckpt = torch.load(old_format_path, map_location="cpu")
    model.projector.load_state_dict(ckpt["projector"])
    model.visual_proj.load_state_dict(ckpt["visual_proj"])
    if "llm_lora" in ckpt:
        model.llm.load_state_dict(ckpt["llm_lora"], strict=False)
    step, val_loss = ckpt.get("step"), ckpt.get("val_loss")
    return (f"old piecewise format: {old_format_path} "
            f"(step={step}, val_loss={val_loss})")


# ── Metrics ──────────────────────────────────────────────────────────────────

def extract_t2_entity(instruction: str) -> Optional[str]:
    """T2 instructions are built as 'Describe the findings related to
    {entity} in this CT scan.' (build_instruction_for_task(),
    aadp/data/ctclip_feature_dataset.py) — pull the entity back out."""
    m = _ENTITY_RE.search(instruction)
    return m.group(1) if m else None


def t3_label_from_target(target: str) -> Optional[bool]:
    """T3 ground truth is 'Yes. <evidence>' / 'No.' — not 'abnormal'/
    'normal'. Returns True/False, or None if the target doesn't parse as
    either (shouldn't happen for real T3 samples, but don't crash if it does)."""
    t = target.strip().lower()
    if t.startswith("yes"):
        return True
    if t.startswith("no"):
        return False
    return None


def t3_prediction_from_generated(generated: str) -> Optional[bool]:
    """First yes/no token in the generated text. None if no clear answer
    (counted as incorrect by the caller, not skipped, same convention as
    qualitative_eval_v2.py's check_yes_no)."""
    m = re.search(r"\b(yes|no)\b", generated.lower())
    if not m:
        return None
    return m.group(1) == "yes"


def rouge_l(scorer, generated: str, target: str) -> float:
    if not target.strip():
        return 0.0
    return scorer.score(target, generated)["rougeL"].fmeasure


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-dir", required=True,
                         help="Directory containing the checkpoint (see module docstring for accepted formats).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--device", default=None,
                         help="Defaults to cuda:<shard-id> under --num-shards>1, else cuda.")
    parser.add_argument("--shard-id", type=int, default=0,
                         help="This process's shard index, 0-based (data-parallel eval: each "
                              "shard independently generates over a disjoint slice of the same "
                              "first --num-samples samples; samples are independent per-row "
                              "generations, so this is embarrassingly parallel — no cross-shard "
                              "communication needed, unlike training's collectives).")
    parser.add_argument("--num-shards", type=int, default=1,
                         help="Total shard count. --output gets a .shardN suffix inserted "
                              "automatically when this is >1; merge with merge_eval_shards.py.")
    args = parser.parse_args()
    assert 0 <= args.shard_id < args.num_shards, "--shard-id must be in [0, --num-shards)"

    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    if args.device is not None:
        device = args.device
    elif args.num_shards > 1:
        # One shard per GPU, GPU index == shard index. Explicit, not derived
        # from SLURM_LOCALID/CUDA_VISIBLE_DEVICES — these processes are
        # plain independent `python3` launches (no torch.distributed, no
        # collectives to keep in sync), so there's no launcher-assigned rank
        # to read; the caller is responsible for launching shard N against a
        # GPU N is allowed to see (see slurm_eval_parallel.sh).
        device = f"cuda:{args.shard_id}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    assert device.startswith("cuda"), "CTViT hardcodes cuda internally — this script requires a GPU."
    # Must happen before any CUDA allocation: CTViT hardcodes bare
    # torch.device('cuda') in several spots internally (not just tensors this
    # script explicitly places with device=device), which resolves to
    # whichever GPU is "current" for the process — set_device() is what
    # actually pins that, the device= kwargs alone are not sufficient. Same
    # bug class as scripts/train_ctclip.py's set_device() fix, and exactly
    # why this exists here too.
    torch.cuda.set_device(torch.device(device))

    print(f"Loading model (llm={cfg.get('llm_model_name')}) on {device}...")
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
        ctclip_checkpoint=cfg.get("ctclip_checkpoint"),
    )

    loaded_desc = load_checkpoint(model, Path(args.checkpoint_dir))
    print(f"Loaded checkpoint — {loaded_desc}")

    # Submodules are constructed/loaded with genuinely different native
    # dtypes (e.g. instruction_encoder loads bf16 by HF default, projector
    # defaults to fp32) — forward()'s `model_dtype = next(self.parameters
    # ()).dtype` only holds a single consistent value under DeepSpeed's
    # blanket bf16 cast, not here. Force one uniform dtype across the whole
    # model, matching qualitative_eval_v2.py's established working pattern.
    model = model.to(device=device, dtype=torch.bfloat16)

    # vector_quantize_pytorch's codebook internally upcasts its distance
    # computation to fp32 (its own @autocast(enabled=False) forward calls
    # .float()) but that only converts the local activation, not the
    # codebook's own stored embedding — so the embedding itself must be
    # kept fp32 too, or the einsum against the upcast activation mismatches.
    # Same restore train_ctclip.py applies after DeepSpeed's blanket bf16 cast.
    codebook = model.visual_encoder.vq._codebook
    codebook.embed.data = codebook.embed.data.float()
    if hasattr(codebook, "cluster_size"):
        codebook.cluster_size.data = codebook.cluster_size.data.float()

    model.eval()

    print(f"Loading val dataset from {cfg['ctrate_csv_valid']}...")
    val_ds = CTRawDataset(
        merged_csv=cfg["ctrate_csv_valid"],
        ct_volume_root=cfg["ct_volume_root"],
        split="val",
        target_shape=tuple(cfg.get("target_shape", [240, 480, 480])),
        task_weights=cfg.get("task_weights", {"T1": 0.5, "T2": 0.35, "T3": 0.15}),
    )
    n = min(args.num_samples, len(val_ds))
    # Interleaved (strided) split of the SAME first-n range, same convention
    # as torch's own DistributedSampler — every shard computes n from the
    # identical len(val_ds)/args.num_samples, so the n-sample pool is
    # identical across shards and the union of all shards' index sets is
    # exactly range(n) with no overlap or gap, regardless of whether n
    # divides evenly by num_shards.
    shard_indices = range(args.shard_id, n, args.num_shards)
    print(f"Evaluating {n} / {len(val_ds)} val samples "
          f"(shard {args.shard_id}/{args.num_shards}: {len(shard_indices)} samples on this process; "
          f"batch_size=1, generation is memory-heavy per volume).")

    val_loader = DataLoader(
        torch.utils.data.Subset(val_ds, shard_indices),
        batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=2,
    )

    rows: List[Dict] = []
    t1_scores, t2_scores = [], []
    t3_correct, t3_scored = 0, 0

    for i, batch in enumerate(val_loader):
        task = batch["task"][0]
        instruction = batch["instruction"][0]
        target = batch["target"][0]
        volume_name = batch["volume_name"][0]

        generated = model.generate(
            instructions=[instruction],
            volume=batch["volume"].to(device=device, dtype=torch.bfloat16),
            max_new_tokens=args.max_new_tokens,
        )[0]

        rows.append({
            "volume_name": volume_name,
            "task": task,
            "instruction": instruction,
            "generated_text": generated,
            "ground_truth": target,
        })

        if task == "T1":
            t1_scores.append(rouge_l(scorer, generated, target))
        elif task == "T2":
            # target is already entity-scoped (build_instruction_for_task()'s
            # T2 branch runs sentences_containing() on the ground-truth
            # report). Scope the generated side the same way before scoring,
            # rather than comparing a possibly full-report generation against
            # an entity-only target — same helper training itself uses, so
            # this can't drift from the actual T2 target semantics.
            entity = extract_t2_entity(instruction)
            scoped_generated = sentences_containing(generated, entity) if entity else generated
            t2_scores.append(rouge_l(scorer, scoped_generated, target))
        elif task == "T3":
            gt_label = t3_label_from_target(target)
            pred_label = t3_prediction_from_generated(generated)
            if gt_label is not None:
                t3_scored += 1
                t3_correct += int(pred_label == gt_label)

        if (i + 1) % 10 == 0 or (i + 1) == len(shard_indices):
            print(f"  [shard {args.shard_id}: {i + 1}/{len(shard_indices)}] "
                  f"{task}  {volume_name}  generated[:80]={generated[:80]!r}")

    out_path = Path(args.output)
    if args.num_shards > 1:
        out_path = out_path.with_suffix(f".shard{args.shard_id}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["volume_name", "task", "instruction", "generated_text", "ground_truth"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")

    t1_mean = sum(t1_scores) / len(t1_scores) if t1_scores else float("nan")
    t2_mean = sum(t2_scores) / len(t2_scores) if t2_scores else float("nan")
    t3_acc = t3_correct / t3_scored if t3_scored else float("nan")

    label = f"shard {args.shard_id}/{args.num_shards} PARTIAL" if args.num_shards > 1 else "Summary"
    print(f"\n=== {label} ===")
    if args.num_shards > 1:
        print("(Run merge_eval_shards.py after all shards finish for the true "
              "aggregate — these per-shard means are NOT directly comparable "
              "to a single-process run's numbers, same reason _validate_e2e() "
              "needed an all_reduce fix in train_ctclip.py: per-shard means "
              "of unequal-n subsets don't average into the correct global mean.)")
    print(f"T1 (n={len(t1_scores)}): mean ROUGE-L = {t1_mean:.4f}")
    print(f"T2 (n={len(t2_scores)}): mean entity-scoped ROUGE-L = {t2_mean:.4f}")
    print(f"T3 (n={t3_scored}): binary accuracy = {t3_acc:.4f} ({t3_correct}/{t3_scored})")


if __name__ == "__main__":
    main()
