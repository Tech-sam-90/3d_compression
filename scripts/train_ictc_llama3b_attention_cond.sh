#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --job-name=ictc_train_llama3b_attn
#SBATCH --output=/scratch/sadeniji/logs/train_llama3b_attn_%j.out
#SBATCH --error=/scratch/sadeniji/logs/train_llama3b_attn_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/ictc_checkpoints_llama3b_attention_cond

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression
python scripts/train_ctclip.py --config configs/ctclip_stage2_llama3b_attention_cond.yaml
