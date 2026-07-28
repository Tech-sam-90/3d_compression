#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke_llama3b_attn
#SBATCH --output=/scratch/sadeniji/logs/smoke_llama3b_attn_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_llama3b_attn_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints_llama3b_attention_cond

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

# Compute nodes have no internet access. All models used here must already
# be cached — without HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE, transformers'
# from_pretrained() still attempts an HTTPS metadata check and hangs
# indefinitely (TCP SYN never gets a response) instead of using the cache.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression
python scripts/train_ctclip.py --config configs/narval_smoke_llama3b_attention_cond.yaml
