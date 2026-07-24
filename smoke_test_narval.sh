#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke
#SBATCH --output=/scratch/sadeniji/logs/smoke_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints

module load gcc arrow/25.0.0
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
python scripts/train_ctclip.py --config configs/narval_smoke.yaml
