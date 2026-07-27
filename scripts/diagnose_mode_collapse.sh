#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=ictc_diagnose
#SBATCH --output=/scratch/sadeniji/logs/diagnose_%j.out
#SBATCH --error=/scratch/sadeniji/logs/diagnose_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

CHECKPOINT="${1:-/scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt}"

python scripts/diagnose_mode_collapse.py \
  --config configs/ctclip_stage2.yaml \
  --checkpoint "$CHECKPOINT"
