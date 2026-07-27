#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke_difflr
#SBATCH --output=/scratch/sadeniji/logs/smoke_difflr_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_difflr_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints_difflr

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

# 200 samples / 2 epochs, as directed — tests whether the differential LR
# fix (aggregator_learning_rate=1e-3 vs learning_rate=2e-5 for LoRA) lets
# the InterSliceAggregator actually differentiate CT scans, without waiting
# for a full 5-epoch run. gradient_accumulation_steps dropped to 1 (from
# the base config's 16): with only 200 samples / batch_size=2 = 100
# batches/epoch, grad_accum=16 would give only 6 real optimizer steps/epoch
# — too few to see whether the fix does anything.
python scripts/train_ctclip.py \
  --config configs/ctclip_stage2.yaml \
  --set max_samples=200 num_epochs=2 gradient_accumulation_steps=1 \
        val_every_n_steps=50 save_every_n_steps=100 patience=99 \
        checkpoint_dir=/scratch/sadeniji/smoke_checkpoints_difflr
