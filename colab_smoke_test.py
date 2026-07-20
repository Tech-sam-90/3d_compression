# colab_smoke_test.py
#
# End-to-end smoke test for the CT-CLIP Stage 2 training pipeline.
# Validates the full path:
#     Google Drive .pt features
#     → CTCLIPFeatureDataset
#     → CTCLIPStage2VLM (BioMedLM backbone)
#     → train_ctclip.py (20-sample run)
#     → checkpoint saved with finite loss
#
# Run cells in order, top to bottom. Fill in HF_TOKEN at the top of CELL 1
# before running anything else.
#
# Required Colab runtime:  GPU — A100 (40 GB).
#   BioMedLM is loaded twice (LLM + instruction encoder): ~22 GB total.
#   Training is only 20 samples (~10 steps) so the whole run takes 25–40 min.
#
# Estimated wall-clock:  ~25–40 min (model download dominates).


# ── CELL 0 — GPU check ────────────────────────────────────────────────────────

import subprocess
import torch

# nvidia-smi summary
result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
     "--format=csv,noheader"],
    capture_output=True, text=True,
)
if result.returncode == 0:
    for line in result.stdout.strip().split("\n"):
        name, total, free = [x.strip() for x in line.split(",")]
        print(f"GPU:  {name}")
        print(f"VRAM: {total} total  |  {free} free")
else:
    print("nvidia-smi not available — are you on a GPU runtime?")

if torch.cuda.is_available():
    idx = torch.cuda.current_device()
    vram_gb = torch.cuda.get_device_properties(idx).total_memory / 1e9
    print(f"\ntorch.cuda: {torch.cuda.get_device_name(idx)}  ({vram_gb:.1f} GB)")
    print(f"torch version: {torch.__version__}")
else:
    print("\nWARNING: torch.cuda.is_available() = False")

print()
print("VRAM requirements for this smoke test (BioMedLM 2.7 B, fp32):")
print("  LLM backbone          ~10.8 GB")
print("  Instruction encoder   ~10.8 GB  (same model, separate instance)")
print("  Projector + activations ~1–2 GB")
print("  ─────────────────────────────")
print("  Total                 ~22–23 GB  →  A100 (40 GB) required")
print()
print("Training is only 20 samples (~10 gradient steps) — expected ~25–40 min total.")


# ── CELL 1 — Install dependencies + clone repo ───────────────────────────────

import os
import subprocess
import sys
from pathlib import Path

# ── USER FILLS IN THESE THREE CONSTANTS ──────────────────────────────────────
GITHUB_REPO = "https://github.com/YOUR_USERNAME/3d-compression.git"  # FILL IN
BRANCH      = "ctclip-stage2-train"
HF_TOKEN    = "hf_..."   # FILL IN — must have accepted CT-RATE gated-dataset terms
# ─────────────────────────────────────────────────────────────────────────────

REPO_DIR = Path("/content/3d-compression")

assert GITHUB_REPO != "https://github.com/YOUR_USERNAME/3d-compression.git", (
    "Replace GITHUB_REPO with your actual repo URL before running CELL 1."
)
assert HF_TOKEN.startswith("hf_") and len(HF_TOKEN) > 10, (
    "Replace HF_TOKEN with your real HuggingFace token before running CELL 1."
)

# Clone repo
if not REPO_DIR.exists():
    print(f"Cloning {GITHUB_REPO}  (branch: {BRANCH}) ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", BRANCH,
         GITHUB_REPO, str(REPO_DIR)],
        check=True,
    )
    print("Clone complete.")
else:
    print(f"{REPO_DIR} already present — skipping clone.")

# Editable install so aadp.* is importable without sys.path hacks
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO_DIR)],
    check=True,
)

# Additional runtime deps
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "nltk", "rouge-score", "sacrebleu", "evaluate", "peft"],
    check=True,
)

# NLTK punkt — needed by vtcb_sweep_ctclip BLEU scorer
import nltk
nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet",   quiet=True)

print("All dependencies installed.")


# ── CELL 2 — Mount Google Drive + configure rclone ───────────────────────────

import subprocess
from pathlib import Path

# Paste your complete [driveB] rclone config block between the triple-quotes.
# This is the same remote used during the CT-CLIP feature extraction run.
# Obtain it with:  rclone config show driveB   on your local machine.
#
# NOTE: do NOT mount Google Drive via google.colab.drive — the feature data
# lives in a different Google account (driveB).  rclone accesses it directly
# using the OAuth token stored in this config block; no Drive mount needed.
RCLONE_CONFIG = """\
[driveB]
# PASTE YOUR [driveB] BLOCK HERE — example layout:
# type = drive
# client_id = ...
# client_secret = ...
# token = {"access_token":"...","token_type":"Bearer","refresh_token":"...","expiry":"..."}
# team_drive =
"""

# Install rclone via apt (fastest on Colab)
subprocess.run(["apt-get", "install", "-y", "-q", "rclone"], check=True)
rclone_ver = subprocess.run(["rclone", "version"], capture_output=True, text=True)
print(rclone_ver.stdout.split("\n")[0])

# Write rclone config
rclone_conf_dir = Path.home() / ".config" / "rclone"
rclone_conf_dir.mkdir(parents=True, exist_ok=True)
rclone_conf_path = rclone_conf_dir / "rclone.conf"
rclone_conf_path.write_text(RCLONE_CONFIG)
print(f"rclone.conf written → {rclone_conf_path}")

# Confirm connectivity — list top-level dirs under ctrate_features/
result = subprocess.run(
    ["rclone", "lsd", "driveB:ctrate_features/"],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print("\nDrive connectivity: OK")
    print(result.stdout.strip() or "  (empty listing — bucket exists but no sub-dirs yet)")
else:
    print("\nDrive connectivity: FAILED")
    print(result.stderr)
    raise RuntimeError(
        "rclone cannot reach driveB:ctrate_features/ — check your config block "
        "and make sure the remote name matches what was used during extraction."
    )


# ── CELL 3 — Copy 20 sample .pt files from Drive ────────────────────────────

import subprocess
import torch
from pathlib import Path

FEATURES_TRAIN_DIR = Path("/content/ctrate_features/train")
FEATURES_VALID_DIR  = Path("/content/ctrate_features/valid")
FEATURES_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_VALID_DIR.mkdir(parents=True, exist_ok=True)

N_TRAIN = 20
N_VALID = 10


def _rclone_copy_first_n(remote_dir: str, local_dir: Path, n: int, label: str) -> None:
    """List .pt files in remote_dir, copy the first n to local_dir."""
    list_result = subprocess.run(
        ["rclone", "lsf", remote_dir, "--include", "*.pt"],
        capture_output=True, text=True,
    )
    if list_result.returncode != 0:
        raise RuntimeError(
            f"rclone lsf failed for {remote_dir}:\n{list_result.stderr}"
        )
    all_files = [f.strip() for f in list_result.stdout.strip().split("\n") if f.strip()]
    if not all_files:
        raise RuntimeError(
            f"No .pt files found in {remote_dir}. "
            "Check that the CT-CLIP extraction pipeline has run and "
            "that the remote path is correct."
        )
    chosen = all_files[:n]
    print(f"  {label}: {len(all_files)} files available, copying first {n}...")

    # Build a rclone filter file so only the chosen files are transferred
    filter_file = Path(f"/tmp/rclone_filter_{label}.txt")
    with open(filter_file, "w") as fh:
        for fname in chosen:
            fh.write(f"+ {fname}\n")
        fh.write("- *\n")

    subprocess.run(
        ["rclone", "copy", remote_dir, str(local_dir),
         "--filter-from", str(filter_file), "--transfers", "4"],
        check=True,
    )
    actual_count = len(list(local_dir.glob("*.pt")))
    print(f"  {label}: {actual_count} .pt files now in {local_dir}")


_rclone_copy_first_n("driveB:ctrate_features/train/", FEATURES_TRAIN_DIR, N_TRAIN, "train")
_rclone_copy_first_n("driveB:ctrate_features/valid/", FEATURES_VALID_DIR,  N_VALID, "valid")

# ── Shape / dtype sanity check ───────────────────────────────────────────────
sample_pt = sorted(FEATURES_TRAIN_DIR.glob("*.pt"))[0]
feat = torch.load(sample_pt, weights_only=True)

EXPECTED_SHAPE = torch.Size([24, 24, 24, 512])
EXPECTED_DTYPE = torch.float16

shape_ok = feat.shape == EXPECTED_SHAPE
dtype_ok = feat.dtype == EXPECTED_DTYPE

print(f"\nSanity check — {sample_pt.name}:")
print(f"  shape : {feat.shape}  (expected {EXPECTED_SHAPE})  {'OK' if shape_ok else 'FAIL'}")
print(f"  dtype : {feat.dtype}  (expected {EXPECTED_DTYPE})  {'OK' if dtype_ok else 'FAIL'}")

if not (shape_ok and dtype_ok):
    raise AssertionError(
        f"Feature tensor shape/dtype mismatch: got {feat.shape} {feat.dtype}, "
        f"expected {EXPECTED_SHAPE} {EXPECTED_DTYPE}."
    )
print("\n  PASSED — feature tensors are correctly shaped and typed.")


# ── CELL 4 — Download CT-RATE CSVs from HuggingFace ─────────────────────────

import pandas as pd
from huggingface_hub import hf_hub_download

HF_REPO_ID       = "ibrahimhamamci/CT-RATE"
TRAIN_CSV_REMOTE = "radiology_text_reports/train_reports.csv"
VALID_CSV_REMOTE = "radiology_text_reports/validation_reports.csv"

print(f"Downloading {TRAIN_CSV_REMOTE} from {HF_REPO_ID} ...")
train_csv_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    filename=TRAIN_CSV_REMOTE,
    token=HF_TOKEN,
)
print(f"Downloading {VALID_CSV_REMOTE} ...")
valid_csv_path = hf_hub_download(
    repo_id=HF_REPO_ID,
    repo_type="dataset",
    filename=VALID_CSV_REMOTE,
    token=HF_TOKEN,
)

train_df = pd.read_csv(train_csv_path)
valid_df = pd.read_csv(valid_csv_path)

print(f"\nTrain CSV : {len(train_df):,} rows  →  {train_csv_path}")
print(f"Valid CSV : {len(valid_df):,} rows  →  {valid_csv_path}")
print(f"Columns   : {list(train_df.columns)}")


# ── CELL 5 — Write smoke-test config ─────────────────────────────────────────

import yaml
from pathlib import Path

CHECKPOINT_DIR = "/content/checkpoints/smoke_test"
CONFIG_PATH    = Path("/content/smoke_test.yaml")

# ── BioMedLM LoRA target modules — architecture note ─────────────────────────
#
# BioMedLM (stanford-crfm/BioMedLM, 2.7 B params) is built on GPT-2 XL.
# GPT-2's self-attention uses a *combined* QKV projection:
#
#     self.c_attn = Conv1D(3 * embed_dim, embed_dim)   ← Q, K, V fused
#     self.c_proj = Conv1D(embed_dim, embed_dim)        ← output projection
#
# This differs from OPT which has separate q_proj / k_proj / v_proj linears.
# For BioMedLM, LoRA must target "c_attn" (the fused QKV Conv1D).
# PEFT ≥ 0.10 handles Conv1D transparently.
#
# DO NOT use ["q_proj", "v_proj"] here — those names don't exist in GPT-2
# and PEFT will raise a ValueError.
#
# BioMedLM hidden_size = 2560.  cond_dim must equal this so the assertion
# in CTCLIPStage2VLM.__init__ passes:
#     assert actual_cond_dim == cond_dim   # 2560 == 2560  ✓
# ─────────────────────────────────────────────────────────────────────────────

smoke_cfg = {
    "experiment_name": "ctclip_stage2_smoke",
    "projector":       "ctclip_stage2",

    # CT-CLIP feature dimensions (fixed — must match extraction pipeline)
    "ctclip_dim": 512,
    "max_depth":  24,

    # Stage 2 architecture
    "embed_dim":  512,
    "num_tokens": 64,
    "num_heads":  8,

    # BioMedLM hidden_size = 2560 — must match instruction_encoder output_dim.
    # CTCLIPStage2VLM.__init__ will assert this; wrong value → AssertionError.
    "cond_dim": 2560,
    "use_film":  True,
    "dropout":   0.0,

    # LLM backbone — BioMedLM (GPT-2 XL architecture, 2.7 B params)
    "llm_model_name": "stanford-crfm/BioMedLM",
    "llm_frozen":     False,
    "llm_lora": {
        "enabled": True,
        "r":       16,
        "alpha":   32,
        # GPT-2 / BioMedLM: use "c_attn" (fused QKV Conv1D).
        # NOT "q_proj" / "v_proj" — those names belong to OPT.
        "target_modules": ["c_attn"],
        "dropout": 0.05,
    },

    # Instruction encoder (same model as LLM — shares tokenizer + architecture)
    "instruction_encoder_model": "stanford-crfm/BioMedLM",

    # Training — smoke-test scale
    "device":                    "cuda",
    "mixed_precision":           True,   # fp16 autocast; GradScaler enabled
    "num_epochs":                1,
    "batch_size":                2,
    "gradient_accumulation_steps": 1,    # effective batch = 2
    "max_grad_norm":             1.0,
    "learning_rate":             1.0e-4,
    "weight_decay":              0.0,
    "warmup_steps":              5,
    "val_every_n_steps":         10,
    "save_every_n_steps":        10,
    "checkpoint_dir":            CHECKPOINT_DIR,
    "patience":                  5,
    "use_wandb":                 False,
    "use_attn_loss":             False,

    # Tasks (T4 excluded — RadGenome unavailable)
    "tasks":        ["T1", "T2", "T3"],
    "task_weights": {"T1": 0.6, "T2": 0.3, "T3": 0.1},

    # Data paths — populated from CELL 3 & 4 outputs
    "features_train_dir": str(FEATURES_TRAIN_DIR),
    "features_valid_dir":  str(FEATURES_VALID_DIR),
    "ctrate_csv_train":   train_csv_path,
    "ctrate_csv_valid":   valid_csv_path,

    "max_samples": 20,       # cap to 20 samples; set null for full dataset
    "hf_token":    HF_TOKEN,
}

CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(CONFIG_PATH, "w") as fh:
    yaml.dump(smoke_cfg, fh, default_flow_style=False, sort_keys=False)

print(f"Smoke-test config written → {CONFIG_PATH}\n")
print(yaml.dump(smoke_cfg, default_flow_style=False, sort_keys=False))


# ── CELL 6 — Run the smoke test ──────────────────────────────────────────────

import subprocess
import sys

TRAIN_SCRIPT = REPO_DIR / "scripts" / "train_ctclip.py"

print(f"Running: python {TRAIN_SCRIPT} --config {CONFIG_PATH}")
print("=" * 70)

# _logged_losses is read by CELL 7 to verify no NaN/Inf during training
_logged_losses = []

proc = subprocess.Popen(
    [sys.executable, str(TRAIN_SCRIPT), "--config", str(CONFIG_PATH)],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    cwd=str(REPO_DIR),   # ensure relative imports inside the script resolve
)

for line in iter(proc.stdout.readline, ""):
    print(line, end="", flush=True)
    # Parse "loss=X.XXXX" tokens so CELL 7 can check for NaN
    if "loss=" in line:
        for token in line.split():
            if token.startswith("loss="):
                try:
                    _logged_losses.append(float(token.split("=")[1]))
                except ValueError:
                    pass

proc.stdout.close()
proc.wait()

print("=" * 70)
if proc.returncode != 0:
    raise RuntimeError(
        f"train_ctclip.py exited with non-zero code {proc.returncode}. "
        "Scroll up for the traceback."
    )
print(f"Training complete. Captured {len(_logged_losses)} loss values for verification.")


# ── CELL 7 — Verify results ──────────────────────────────────────────────────

import math
import torch
from pathlib import Path

PASSED   = True
findings = []


# ── Check 1: at least one checkpoint file exists ─────────────────────────────
ckpt_dir   = Path(CHECKPOINT_DIR)
all_ckpts  = sorted(ckpt_dir.glob("*.pt"))

if not all_ckpts:
    PASSED = False
    findings.append("FAIL  No .pt checkpoint files found in checkpoint_dir.")
else:
    findings.append(f"OK    {len(all_ckpts)} checkpoint(s): {[c.name for c in all_ckpts]}")


# ── Check 2: load the latest/best checkpoint and inspect keys ─────────────────
ckpt      = None
ckpt_file = (ckpt_dir / "checkpoint_latest.pt") if (ckpt_dir / "checkpoint_latest.pt").exists() \
            else (ckpt_dir / "checkpoint_best.pt") if (ckpt_dir / "checkpoint_best.pt").exists() \
            else (all_ckpts[-1] if all_ckpts else None)

if ckpt_file is None:
    PASSED = False
    findings.append("FAIL  No checkpoint could be identified for loading.")
else:
    try:
        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=True)
        findings.append(f"OK    Loaded: {ckpt_file.name}")
        findings.append(f"      Keys:   {sorted(ckpt.keys())}")
        findings.append(f"      Step:   {ckpt.get('step', 'N/A')}")
        findings.append(f"      Epoch:  {ckpt.get('epoch', 'N/A')}")
        findings.append(f"      val_loss: {ckpt.get('val_loss', 'N/A')}")
    except Exception as exc:
        PASSED = False
        findings.append(f"FAIL  Could not load checkpoint: {exc}")


# ── Check 3: required keys present in checkpoint ─────────────────────────────
if ckpt is not None:
    required_keys = {"projector", "visual_proj", "optimizer", "scheduler", "step"}
    missing_keys  = required_keys - set(ckpt.keys())
    if missing_keys:
        PASSED = False
        findings.append(f"FAIL  Checkpoint missing expected keys: {missing_keys}")
    else:
        findings.append(f"OK    All required checkpoint keys present.")


# ── Check 4: all logged training losses are finite (no NaN / Inf) ─────────────
if _logged_losses:
    bad = [v for v in _logged_losses if not math.isfinite(v)]
    if bad:
        PASSED = False
        findings.append(
            f"FAIL  {len(bad)} non-finite loss value(s) detected: {bad[:5]}..."
        )
    else:
        findings.append(
            f"OK    All {len(_logged_losses)} logged losses are finite "
            f"(min={min(_logged_losses):.4f}, max={max(_logged_losses):.4f})."
        )
else:
    # Not a hard failure — the log format might differ
    findings.append(
        "WARN  No loss values were parsed from training output. "
        "Scroll up and verify manually that losses printed without 'NaN'."
    )


# ── Check 5: checkpoint val_loss is finite ────────────────────────────────────
if ckpt is not None:
    val_loss = ckpt.get("val_loss", None)
    if val_loss is not None:
        if not math.isfinite(float(val_loss)):
            PASSED = False
            findings.append(f"FAIL  Checkpoint val_loss={val_loss} is not finite.")
        else:
            findings.append(f"OK    Checkpoint val_loss={float(val_loss):.4f} is finite.")


# ── Print findings ────────────────────────────────────────────────────────────
print("─" * 70)
print("Verification findings:")
for f in findings:
    print(f"  {f}")
print("─" * 70)

if PASSED:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                      SMOKE TEST PASSED  ✓                           ║
╚══════════════════════════════════════════════════════════════════════╝
""")
else:
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                      SMOKE TEST FAILED  ✗                           ║
║  Review the FAIL lines above and the training output in CELL 6.     ║
╚══════════════════════════════════════════════════════════════════════╝
""")


# ── Next steps on HPC ────────────────────────────────────────────────────────
print("Next steps — full production training on HPC / Colab A100:")
print()
print("  1. Scale up batch / accumulation  (already on BioMedLM + A100)")
print("       batch_size:                  8")
print("       gradient_accumulation_steps: 4   # effective batch = 32")
print()
print("  3. Full training duration")
print("       num_epochs:                  5")
print("       max_samples:                 null  # remove cap, use all 50,188 volumes")
print()
print("  4. Fill in HPC data paths in configs/ctclip_stage2.yaml")
print("       features_train_dir:  /path/to/ctrate_features/train")
print("       features_valid_dir:  /path/to/ctrate_features/valid")
print("       ctrate_csv_train:    /path/to/CT-RATE/train_reports.csv")
print("       ctrate_csv_valid:    /path/to/CT-RATE/validation_reports.csv")
print()
print("  5. Enable W&B logging")
print("       use_wandb:           true")
print()
print("  6. Launch")
print("       python scripts/train_ctclip.py --config configs/ctclip_stage2.yaml")
