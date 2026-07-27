#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
# Same mem/cpus allocation as smoke_test_narval.sh (BioMedLM at max_length=1024
# OOM'd host RAM under lighter allocations — job 66394290).
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke_attncond
#SBATCH --output=/scratch/sadeniji/logs/smoke_attncond_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_attncond_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints_attention_cond

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression
python scripts/train_ctclip.py --config configs/narval_smoke_attention_cond.yaml
