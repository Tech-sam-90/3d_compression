#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=ictc_vtcb_sweep
#SBATCH --output=/scratch/sadeniji/logs/vtcb_sweep_%j.out
#SBATCH --error=/scratch/sadeniji/logs/vtcb_sweep_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p results/vtcb_sweep_ctclip

# apptainer is needed on PATH for the RadGraph-XL metric — it runs inside a
# separate pinned container (see aadp/evaluation/metrics/container_bridge.py
# and scripts/build_metrics_container.sh) since radgraph==0.1.18 needs HF
# tokenizer APIs this venv's transformers==5.14.1 removed.
module load gcc arrow/25.0.0 apptainer/1.3.5
source ~/envs/ictc/bin/activate

# Compute nodes have no internet access. All models used here must already
# be cached — without HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE, transformers'
# from_pretrained() still attempts an HTTPS metadata check and hangs
# indefinitely (TCP SYN never gets a response) instead of using the cache.
# The three HF_HOME/TRANSFORMERS_CACHE/HF_DATASETS_CACHE vars point at
# persistent project storage (not $HOME) — populate it first by running
# `python scripts/cache_all_models.py --hf_token ...` on the LOGIN node.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

# Sweeps vtcb_budgets from configs/ctclip_stage2.yaml (default
# [16, 32, 64, 128, 256, 512]): for each M, rebuilds Stage 2 at that budget,
# fine-tunes 1 epoch (lr=1e-5) from the checkpoint below, generates reports
# on the full val set, and scores all 8 Argus Table 2 metrics via
# aadp/evaluation/metrics/compute_all.py (6 real; GREEN stubbed NaN — see
# that file's comment; RadGraph-XL real via the container above). Saves
# results/vtcb_sweep_ctclip/M={M}_scores.json per budget plus a summary.
#
# --checkpoint must point at a trained checkpoint_best.pt from train_ictc.sh
# (checkpoint_dir in configs/ctclip_stage2.yaml). Override on the command
# line if training saved elsewhere or you want a different checkpoint:
#   sbatch vtcb_sweep.sh /path/to/checkpoint_best.pt
CHECKPOINT="${1:-/scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt}"

python scripts/vtcb_sweep_ctclip.py \
  --config configs/ctclip_stage2.yaml \
  --checkpoint "$CHECKPOINT"
