#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_eval_checkpoint
#SBATCH --output=/scratch/sadeniji/logs/eval_checkpoint_%j.out
#SBATCH --error=/scratch/sadeniji/logs/eval_checkpoint_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0 apptainer/1.3.5
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

CHECKPOINT="${1:-/scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt}"
CONFIG="${2:-configs/ctclip_stage2.yaml}"

python scripts/evaluate_checkpoint.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --max_samples 200 \
  --n_examples 5
