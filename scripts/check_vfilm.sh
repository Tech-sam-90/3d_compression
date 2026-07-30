#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=ictc_check_vfilm
#SBATCH --output=/scratch/sadeniji/logs/check_vfilm_%j.out
#SBATCH --error=/scratch/sadeniji/logs/check_vfilm_%j.err

mkdir -p /scratch/sadeniji/logs

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

python scripts/check_vfilm.py \
  --ckpt /scratch/sadeniji/ictc_checkpoints_llama3b_v2/checkpoint_best.pt \
  --config configs/ctclip_stage2_llama3b_v2.yaml \
  --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid
