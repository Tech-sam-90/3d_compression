#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=14:00:00
#SBATCH --job-name=ictc_t2t3_fix
#SBATCH --output=/scratch/sadeniji/logs/t2t3_fix_%j.out
#SBATCH --error=/scratch/sadeniji/logs/t2t3_fix_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/ictc_checkpoints_t2t3_fix

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

# Compute nodes have no internet access — without HF_HUB_OFFLINE/
# TRANSFORMERS_OFFLINE, transformers' from_pretrained() still attempts an
# HTTPS metadata check and hangs indefinitely instead of using the cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

python scripts/train_ctclip.py --config configs/ctclip_t2t3_fix.yaml
