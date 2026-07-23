#!/bin/bash
# build_metrics_container.sh — build the pinned Apptainer container that runs
# GREEN and RadGraph-XL for real (non-NaN) scores.
#
# WHY a separate container: GREEN (green-score) and RadGraph-XL (radgraph)
# were both built against transformers==4.40.0 / torch==2.2.2 (~2024). Our
# main training venv (~/envs/ictc) runs transformers==5.14.1, and both
# packages break there in different ways — see the "KNOWN LIMITATION" notes
# in aadp/evaluation/metrics/green.py and radgraph_f1.py. Separately, this
# cluster's Python builds report a bare `linux_x86_64` platform tag instead
# of a standard `manylinux*` tag, so real PyPI wheels for compiled packages
# (pyarrow, etc.) can't be installed into ANY venv here — confirmed via
# extensive testing. A container sidesteps both problems: standard Python,
# standard platform tags, and we pin torch/transformers to versions that
# predate the removed tokenizer APIs radgraph's vendored AllenNLP code needs.
#
# radgraph's actual requirements (torch>=2.1.0, transformers>=4.39.0, no
# upper bound) are compatible with GREEN's exact pins, so ONE container
# serves both metrics.
#
# Run this on the Narval LOGIN NODE (has internet). Takes a while — pulls a
# ~3.7GB CUDA image, installs ~2GB of Python packages, downloads GREEN's 7B
# model (~10GB). Do NOT run the actual GREEN/RadGraph-XL inference on the
# login node — loading a 7B model there gets SIGKILLed by the login node's
# memory guardrail (confirmed). Inference must run on a GPU compute node
# via `apptainer exec --nv` (see aadp/evaluation/metrics/container_bridge.py
# and scripts/container_metrics_worker.py).
#
# Usage:
#   module load apptainer/1.3.5   # if not already loaded
#   bash scripts/build_metrics_container.sh

set -euo pipefail

CONTAINER_DIR=/project/def-uanazodo-ab/sadeniji/containers
SITE_DIR=/project/def-uanazodo-ab/sadeniji/container_site
HF_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
SIF="$CONTAINER_DIR/pytorch_2.2.2_cu121.sif"
GREEN_SRC="$CONTAINER_DIR/GREEN_src"

mkdir -p "$CONTAINER_DIR" "$SITE_DIR" "$HF_CACHE"
export APPTAINER_CACHEDIR="$CONTAINER_DIR/.apptainer_cache"
mkdir -p "$APPTAINER_CACHEDIR"

# ── 1. Base image (torch==2.2.2, CUDA 12.1, real manylinux platform tags) ──
if [ ! -f "$SIF" ]; then
  echo "[1/4] Pulling pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime..."
  apptainer pull "$SIF" docker://pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime
else
  echo "[1/4] $SIF already exists, skipping pull."
fi

# ── 2. GREEN source, patched only for python_requires ──────────────────────
# torch/transformers pins are kept EXACTLY as upstream (the base image
# already ships torch==2.2.2 natively). torchvision/opencv-python are
# dropped — grep of green_score/*.py shows neither is actually imported,
# only listed in setup.py (dead weight even upstream). python_requires is
# relaxed from the exact "==3.12.1" upstream pin to match this image's
# Python 3.10.14 (one patch version above this repo's own former ceiling,
# see the commented-out line in the file) — nothing in the source is
# 3.12-specific.
if [ ! -d "$GREEN_SRC" ]; then
  echo "[2/4] Cloning Stanford-AIMI/GREEN..."
  git clone --depth 1 https://github.com/Stanford-AIMI/GREEN.git "$GREEN_SRC"
  python3 - "$GREEN_SRC/setup.py" <<'PYEOF'
import re, sys
path = sys.argv[1]
src = open(path).read()
src = src.replace('        "torchvision==0.17.2",\n', "")
src = src.replace('        "opencv-python==4.10.0.84",\n', "")
src = src.replace('python_requires="==3.12.1"', 'python_requires=">=3.9,<3.13"')
open(path, "w").write(src)
PYEOF
else
  echo "[2/4] $GREEN_SRC already exists, skipping clone."
fi

# ── 3. Install GREEN + radgraph into a host-side target dir ────────────────
# Installed to a bind-mounted host directory (not baked into the read-only
# .sif) so no fakeroot/sandbox-build privileges are needed — a standard,
# low-risk Apptainer pattern. --cleanenv avoids this host's
# SSL_CERT_FILE/CURL_CA_BUNDLE (pointed at a RedHat-style path) leaking into
# the Ubuntu-based container and breaking pip's TLS.
echo "[3/4] Installing green-score + radgraph into $SITE_DIR..."
apptainer exec --cleanenv \
  --bind "$SITE_DIR:/container_site" \
  --bind "$GREEN_SRC:/green_src" \
  "$SIF" \
  python -m pip install --target=/container_site \
  transformers==4.40.0 accelerate==0.30.1 pillow==10.3.0 sentencepiece==0.2.0 \
  sentence-transformers==3.0.1 datasets==3.2.0 dill==0.3.8 protobuf==5.29.1 \
  scipy matplotlib scikit-learn pandas "numpy<2" \
  radgraph==0.1.18 /green_src

# ── 4. Pre-download models (login node has internet; compute nodes don't) ──
echo "[4/4] Pre-downloading RadGraph-XL and GREEN models to $HF_CACHE..."
apptainer exec --cleanenv --nv \
  --bind "$SITE_DIR:/container_site" \
  --env PYTHONPATH=/container_site \
  --env HF_HOME="$HF_CACHE" \
  --env TRANSFORMERS_CACHE="$HF_CACHE" \
  --env HF_DATASETS_CACHE="$HF_CACHE" \
  --env HF_TOKEN="${HF_TOKEN:-}" \
  "$SIF" \
  python -c "
from radgraph import F1RadGraph
print('Downloading radgraph-xl...')
F1RadGraph(reward_level='all', model_type='radgraph-xl')
print('radgraph-xl OK')

from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'StanfordAIMI/GREEN-radllama2-7b'
print('Downloading GREEN tokenizer...')
AutoTokenizer.from_pretrained(name, trust_remote_code=True)
print('Downloading GREEN model weights (this does NOT load them into memory)...')
from huggingface_hub import snapshot_download
snapshot_download(name)
print('GREEN OK')
"

echo "Done. Container: $SIF"
echo "Package site:    $SITE_DIR"
du -sh "$SIF" "$SITE_DIR" "$HF_CACHE" 2>/dev/null || true
