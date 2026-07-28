#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
# Matches train_ictc_llama3b.sh's allocation — Llama-3.2-3B-Instruct loaded
# twice (LLM + instruction encoder, no cond_dim projection layer to allow a
# smaller encoder), both in fp32.
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=0:45:00
#SBATCH --job-name=ictc_test3_llama3b
#SBATCH --output=/scratch/sadeniji/logs/test3_llama3b_%j.out
#SBATCH --error=/scratch/sadeniji/logs/test3_llama3b_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression
python3 scripts/diagnostics/test3_full_rerun_llama3b.py
