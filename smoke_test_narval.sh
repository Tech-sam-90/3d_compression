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

# Compute nodes have no internet access. All models used here are already
# cached under ~/.cache/huggingface — without these, transformers'
# from_pretrained() still attempts an HTTPS metadata check and hangs
# indefinitely (TCP SYN never gets a response) instead of using the cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd ~/3d_compression
python scripts/train_ctclip.py --config configs/narval_smoke.yaml
