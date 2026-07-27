#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
# mem/cpus-per-task match train_ictc_stage2.sh's proven allocation for
# BioMedLM (2.7B) at max_length=1024. This script originally pointed at a
# lightweight OPT-125m joint-smoke config (mem=40G/cpus=4 was fine there);
# job 66394290 OOM'd (host RAM, SLURM oom_kill, exit 125) a few real
# optimizer steps into the real Stage 2 forward pass under the old values.
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke_stage2
#SBATCH --output=/scratch/sadeniji/logs/smoke_stage2_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_stage2_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints_stage2

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
python scripts/train_ctclip.py --config configs/narval_smoke_stage2.yaml
