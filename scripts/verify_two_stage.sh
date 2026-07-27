#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=ictc_verify_2stage
#SBATCH --output=/scratch/sadeniji/logs/verify_2stage_%j.out
#SBATCH --error=/scratch/sadeniji/logs/verify_2stage_%j.err

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

# Full suite: the two-stage training changes touch CTCLIPStage2VLM.forward()
# (new cls_head + compute_lm gate) and train_ctclip.py broadly, not just the
# stage2 projector, so a targeted subset isn't enough this time.
# (notebooks/colab_smoke_test.py is a standalone Colab script, not a test
# module — asserts at import time if collected from the repo root, so scope
# to tests/ explicitly rather than bare `pytest`.)
pytest tests/ -v
