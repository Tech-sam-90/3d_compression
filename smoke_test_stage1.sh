#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke_stage1
#SBATCH --output=/scratch/sadeniji/logs/smoke_stage1_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_stage1_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints_stage1

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

# 200 samples / 2 epochs — verify cls_loss decreases and the aggregator's
# LoRA-free training actually runs (correct freezing, cls_head wired up,
# no LLM forward invoked) before committing to the full Stage 1 run.
python scripts/train_ctclip.py \
  --config configs/ctclip_stage1.yaml \
  --set max_samples=200 num_epochs=2 gradient_accumulation_steps=1 \
        val_every_n_steps=50 save_every_n_steps=100 patience=99 \
        checkpoint_dir=/scratch/sadeniji/smoke_checkpoints_stage1
