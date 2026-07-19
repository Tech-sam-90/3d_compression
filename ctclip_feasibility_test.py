# ctclip_feasibility_test.py
#
# GO/NO-GO feasibility test for pre-extracting CT-CLIP (CTViT) visual
# features, before committing to building a full extraction pipeline for
# ICTC Stage 2.
#
# BEFORE COMMITTING ANY CHANGES FROM THIS SCRIPT, make sure you're on a
# feature branch so main is never touched:
#
#     git checkout -b ctclip-stage2 && git push -u origin ctclip-stage2
#
# Each "# ── CELL N ──" section below is self-contained enough to paste as
# its own Colab cell, run top to bottom in order.
#
# Requires: HF_TOKEN env var set to a token that has accepted the CT-RATE
# gated-dataset terms on HuggingFace.

# ── CELL 1 — Config (hyperparameters as named constants) ───────────────────

import os
from pathlib import Path

N_VOLUMES = 10
HF_REPO = "ibrahimhamamci/CT-RATE"
CT_CLIP_REPO_URL = "https://github.com/ibrahimethemhamamci/CT-CLIP.git"
CT_CLIP_LOCAL_DIR = Path("./CT-CLIP")
WORK_DIR = Path("./ctclip_feasibility_work")
WEIGHTS_PATH = WORK_DIR / "CT-CLIP_v2.pt"
FEATURES_DIR = WORK_DIR / "features"

# Target preprocessing shape, per CT-CLIP's own data.py: (H, W, D).
TARGET_SHAPE = (480, 480, 240)

# CTViT / CTCLIP construction config, from CT-CLIP's scripts/run_zero_shot.py.
CTVIT_CONFIG = dict(
    dim=512,
    codebook_size=8192,
    image_size=480,
    patch_size=20,
    temporal_patch_size=10,
    spatial_depth=4,
    temporal_depth=4,
    dim_head=32,
    heads=8,
)
CTCLIP_CONFIG = dict(
    dim_image=294912,
    dim_text=768,
    dim_latent=512,
    extra_latent_projection=False,
    use_mlm=False,
    downsample_image_embeds=False,
    use_all_token_embeds=False,
)
TEXT_ENCODER_NAME = "microsoft/BiomedVLP-CXR-BERT-specialized"

# HF glob pattern for finding real volumes to stream.
# NOTE: this "data/" prefix is what was specified for this script, but
# earlier in this project a real HfApi query (scripts/check_dataset_size.py)
# confirmed this repo's actual v2 layout uses "dataset/train_fixed/..."
# instead. If the glob below returns 0 files, that's almost certainly why -
# try "datasets/{HF_REPO}/dataset/train_fixed/**/*.nii.gz" instead.
HF_GLOB_PATTERN = f"datasets/{HF_REPO}/data/**/*.nii.gz"

# If set to a list of local .nii.gz paths, skip the HuggingFace streaming
# step entirely and use these instead.
LOCAL_NII_FILES: "list[str] | None" = None

# Google Drive pricing tiers to check the extrapolated storage against.
DRIVE_TIERS_GB = {
    "15 GB (free)": 15,
    "100 GB (~$2/mo)": 100,
    "2 TB (~$10/mo)": 2048,
}
FULL_DATASET_VOLUMES = 25_000

WORK_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)


# ── CELL 2 — Install dependencies ───────────────────────────────────────────

import subprocess
import sys


def _pip_install(*packages: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)


_pip_install("nibabel", "huggingface_hub", "transformers", "torch")

if not CT_CLIP_LOCAL_DIR.exists():
    subprocess.run(["git", "clone", CT_CLIP_REPO_URL, str(CT_CLIP_LOCAL_DIR)], check=True)
else:
    print(f"{CT_CLIP_LOCAL_DIR} already exists, skipping clone.")

# CT-CLIP has NO setup.py at its repo root. transformer_maskgit (CTViT) and
# CT_CLIP (the CTCLIP wrapper) are each their own installable subdirectory,
# each nesting the real package one level deeper
# (CT-CLIP/transformer_maskgit/transformer_maskgit/ctvit.py). Install both
# separately, and do NOT put the repo root on sys.path: the outer
# CT-CLIP/transformer_maskgit/ folder has no __init__.py, so it would be
# picked up as an empty namespace package that shadows the real install
# ("cannot import name 'CTViT' from 'transformer_maskgit' (unknown location)").
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e",
     str(CT_CLIP_LOCAL_DIR / "transformer_maskgit")],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-e",
     str(CT_CLIP_LOCAL_DIR / "CT_CLIP")],
    check=True,
)
print("Dependencies installed.")
print("In Colab: RESTART THE RUNTIME now, then continue from CELL 3 - a failed "
      "import caches a broken module in sys.modules that a re-run won't clear.")


# ── CELL 3 — Download CT-CLIP_v2.pt weights ─────────────────────────────────

HF_TOKEN = os.environ.get("HF_TOKEN", "")
assert HF_TOKEN, "Set HF_TOKEN to a token that has accepted the CT-RATE gated dataset terms."

if WEIGHTS_PATH.exists():
    print(f"Weights already downloaded at {WEIGHTS_PATH}, skipping.")
else:
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    all_files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset", token=HF_TOKEN)
    candidates = [f for f in all_files if "CT-CLIP_v2" in f and f.endswith(".pt")]
    if not candidates:
        raise FileNotFoundError(
            f"No file matching 'CT-CLIP_v2*.pt' found in {HF_REPO}. Browse "
            f"https://huggingface.co/datasets/{HF_REPO}/tree/main to find the "
            "real path and hardcode it here."
        )
    remote_path = candidates[0]
    print(f"Found weights at: {remote_path}")

    local_path = Path(
        hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            filename=remote_path,
            token=HF_TOKEN,
            local_dir=str(WORK_DIR),
        )
    )
    if local_path != WEIGHTS_PATH:
        local_path.rename(WEIGHTS_PATH)

print(f"Weights ready at {WEIGHTS_PATH}")


# ── CELL 4 — Build CTCLIP model ──────────────────────────────────────────────

import torch
from transformer_maskgit import CTViT
from transformers import BertModel, BertTokenizer

try:
    from ct_clip import CTCLIP
except ImportError as exc:
    raise ImportError(
        "Could not import CTCLIP from the ct_clip package - check "
        "CT-CLIP/scripts/run_zero_shot.py in the cloned repo for the current "
        "correct import path if the package layout has changed."
    ) from exc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

text_tokenizer = BertTokenizer.from_pretrained(TEXT_ENCODER_NAME)
text_encoder = BertModel.from_pretrained(TEXT_ENCODER_NAME)
# Text encoder is required for CTCLIP's __init__ but not used at inference -
# only clip.visual_transformer(...) is called below.

image_encoder = CTViT(**CTVIT_CONFIG)

clip = CTCLIP(
    image_encoder=image_encoder,
    text_encoder=text_encoder,
    tokenizer=text_tokenizer,
    **CTCLIP_CONFIG,
)

# Don't use clip.load(): it does a strict load_state_dict, which rejects the
# whole checkpoint over version drift. transformers >=4.31 removed the
# `position_ids` buffer from BertModel's embeddings, but CT-CLIP_v2.pt was
# saved when it still existed. It's a non-learned arange() buffer, so dropping
# it loses nothing.
state_dict = torch.load(str(WEIGHTS_PATH), map_location="cpu")
state_dict.pop("text_transformer.embeddings.position_ids", None)

# Load non-strictly, but REPORT what didn't match. A silently-missing
# visual_transformer.* key would leave that part of the encoder randomly
# initialised and make every feature this script extracts meaningless.
missing, unexpected = clip.load_state_dict(state_dict, strict=False)
if missing:
    print(f"WARNING - {len(missing)} missing key(s), first few: {missing[:5]}")
if unexpected:
    print(f"WARNING - {len(unexpected)} unexpected key(s), first few: {unexpected[:5]}")
if not missing and not unexpected:
    print("All checkpoint keys matched exactly.")

visual_missing = [k for k in missing if k.startswith("visual_transformer")]
if visual_missing:
    raise RuntimeError(
        f"{len(visual_missing)} visual_transformer key(s) missing from the "
        f"checkpoint (e.g. {visual_missing[:3]}). The visual encoder would be "
        "partly random and every extracted feature meaningless - fix this "
        "before trusting any number from this script."
    )

clip = clip.to(DEVICE)
clip.eval()
for p in clip.parameters():
    p.requires_grad_(False)
print("CTCLIP model built and weights loaded.")


# ── CELL 5 — Get 10 real volumes (HF stream, or LOCAL_NII_FILES) ────────────

raw_dir = WORK_DIR / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)
volume_paths: list[tuple[str, Path]] = []

if LOCAL_NII_FILES:
    print(f"Using {len(LOCAL_NII_FILES)} volumes from LOCAL_NII_FILES.")
    volume_paths = [(Path(p).name, Path(p)) for p in LOCAL_NII_FILES[:N_VOLUMES]]
else:
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem(token=HF_TOKEN)
    print(f"Globbing {HF_GLOB_PATTERN} (walks the real repo tree, may take a moment) ...")
    all_paths = fs.glob(HF_GLOB_PATTERN)
    if not all_paths:
        raise FileNotFoundError(
            f"No .nii.gz files matched '{HF_GLOB_PATTERN}'. This repo's real v2 "
            f"layout was confirmed earlier in this project to use "
            f"'datasets/{HF_REPO}/dataset/train_fixed/**/*.nii.gz' instead - try that."
        )
    chosen = sorted(all_paths)[:N_VOLUMES]
    for hf_path in chosen:
        name = Path(hf_path).name
        local_path = raw_dir / name
        if not local_path.exists():
            print(f"  Streaming {name} ...")
            with fs.open(hf_path, "rb") as src, open(local_path, "wb") as dst:
                dst.write(src.read())
        volume_paths.append((name, local_path))

print(f"Got {len(volume_paths)} volumes.")


# ── CELL 6 — Preprocessing (matches CT-CLIP's data.py exactly) ─────────────

import nibabel as nib
import numpy as np


def center_crop_or_pad(arr: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Crop (centered) or zero-pad (centered) each axis independently to
    reach target_shape. Operates on an (H, W, D)-ordered array."""
    out = arr
    for axis, target in enumerate(target_shape):
        cur = out.shape[axis]
        if cur > target:
            start = (cur - target) // 2
            idx = [slice(None)] * out.ndim
            idx[axis] = slice(start, start + target)
            out = out[tuple(idx)]
        elif cur < target:
            pad_total = target - cur
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            pad_width = [(0, 0)] * out.ndim
            pad_width[axis] = (pad_before, pad_after)
            out = np.pad(out, pad_width, mode="constant", constant_values=0.0)
    return out


def load_and_preprocess(path: Path) -> torch.Tensor:
    img = nib.load(str(path))
    arr = np.asarray(img.dataobj, dtype=np.float32)  # raw NIfTI axis order: (H, W, D)

    arr = np.clip(arr, -1000.0, 1000.0) / 1000.0  # HU clip -> [-1, 1]
    arr = center_crop_or_pad(arr, TARGET_SHAPE)  # (480, 480, 240)

    arr = arr.transpose(2, 0, 1)  # (H, W, D) -> (D, H, W)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 240, 480, 480)
    return tensor.float()


# ── CELL 7 — Run extraction, measure, save ──────────────────────────────────

import time

rows = []
saved_pt_paths = []

for i, (name, local_path) in enumerate(volume_paths):
    raw_bytes = local_path.stat().st_size

    volume_tensor = load_and_preprocess(local_path).to(DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        features = clip.visual_transformer(volume_tensor, return_encoded_tokens=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    infer_seconds = time.perf_counter() - t0

    features = features.squeeze(0).half()  # (1, 24, 24, 24, 512) -> (24, 24, 24, 512)
    out_path = FEATURES_DIR / f"vol_{i:03d}.pt"
    torch.save(features, out_path)
    feature_bytes = out_path.stat().st_size
    saved_pt_paths.append(out_path)

    rows.append(
        {
            "name": name,
            "raw_mb": raw_bytes / 1e6,
            "feature_mb": feature_bytes / 1e6,
            "infer_s": infer_seconds,
            "shape": tuple(features.shape),
        }
    )
    print(
        f"  {name:30s} raw={rows[-1]['raw_mb']:7.1f} MB  "
        f"feat={rows[-1]['feature_mb']:6.2f} MB  "
        f"infer={infer_seconds:5.2f} s  shape={rows[-1]['shape']}"
    )


# ── CELL 8 — Summary table + GO/NO-GO verdict ───────────────────────────────

print("\nPer-volume summary")
print("-" * 80)
print(f"{'volume':30s} {'raw MB':>10s} {'feature MB':>12s} {'infer s':>10s}")
for r in rows:
    print(f"{r['name']:30s} {r['raw_mb']:10.1f} {r['feature_mb']:12.2f} {r['infer_s']:10.2f}")
print("-" * 80)

n = len(rows)
avg_raw_mb = sum(r["raw_mb"] for r in rows) / n
avg_feature_mb = sum(r["feature_mb"] for r in rows) / n
avg_infer_s = sum(r["infer_s"] for r in rows) / n
token_shape = rows[0]["shape"]
n_tokens = token_shape[0] * token_shape[1] * token_shape[2]

print(f"Feature shape: {token_shape}  ({n_tokens:,} tokens/volume)")
print(f"Averages over {n} real volumes:")
print(f"  raw size:     {avg_raw_mb:.1f} MB/volume")
print(f"  feature size: {avg_feature_mb:.2f} MB/volume (fp16)")
print(f"  compression ratio (raw/feature): {avg_raw_mb / avg_feature_mb:.2f}x")
print(f"  inference:    {avg_infer_s:.2f} s/volume")

total_gb = avg_feature_mb * FULL_DATASET_VOLUMES / 1024
total_hours = avg_infer_s * FULL_DATASET_VOLUMES / 3600

print(f"\nExtrapolated to the full {FULL_DATASET_VOLUMES:,}-volume CT-RATE training set:")
print(f"  total feature storage: ~{total_gb:.1f} GB")
print(f"  total extraction time (single GPU): ~{total_hours:.1f} hours")

print("\nAgainst Google Drive tiers:")
fitting_tier = None
for tier_name, tier_gb in DRIVE_TIERS_GB.items():
    fits = total_gb <= tier_gb
    print(f"  {tier_name:20s} ({tier_gb:>5} GB): {'fits' if fits else 'does NOT fit'}")
    if fits and fitting_tier is None:
        fitting_tier = tier_name

if fitting_tier is not None:
    print(f"\nVERDICT: GO - fits in the {fitting_tier} plan (~{total_gb:.1f} GB needed).")
else:
    print(
        f"\nVERDICT: NO-GO - ~{total_gb:.1f} GB exceeds every tier checked "
        f"(largest: {max(DRIVE_TIERS_GB.values())} GB)."
    )
    print(
        "Fallback recommendation: don't pre-extract to disk. Stream CT-CLIP "
        "features from HuggingFace at training time inside the DataLoader "
        "instead (encode each volume on the fly)."
    )


# ── CELL 9 — Sanity-check vol_000.pt ─────────────────────────────────────────

check_path = FEATURES_DIR / "vol_000.pt"
saved = torch.load(check_path, map_location="cpu")
saved_f32 = saved.float()

print(f"\nSanity check: {check_path}")
print(f"  shape: {tuple(saved.shape)}")
print(f"  dtype: {saved.dtype}")
print(f"  mean:  {saved_f32.mean().item():.4f}")
print(f"  std:   {saved_f32.std().item():.4f}")
print(f"  min:   {saved_f32.min().item():.4f}")
print(f"  max:   {saved_f32.max().item():.4f}")
