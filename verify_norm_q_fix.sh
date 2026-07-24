#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --job-name=ictc_verify_normq
#SBATCH --output=/scratch/sadeniji/logs/verify_normq_%j.out
#SBATCH --error=/scratch/sadeniji/logs/verify_normq_%j.err

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

# Sanity-check the norm_q removal in InterSliceAggregator before committing
# to a 48-hour run: forward/backward shape + gradient-flow tests only,
# not a full suite run.
pytest tests/test_stage2.py tests/test_ctclip_stage2_vlm.py -v
