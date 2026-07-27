#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_smoke_biomedlm
#SBATCH --output=/scratch/sadeniji/logs/smoke_biomedlm_%j.out
#SBATCH --error=/scratch/sadeniji/logs/smoke_biomedlm_%j.err

mkdir -p /scratch/sadeniji/logs
mkdir -p /scratch/sadeniji/smoke_checkpoints_biomedlm

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

# Unlike smoke_test_narval.sh (configs/narval_smoke.yaml, facebook/opt-125m
# — a different model, different LoRA target_modules), this exercises the
# ACTUAL configs/ctclip_stage2.yaml (BioMedLM, target_modules=[c_attn,
# c_proj], learning_rate=5e-5) at a tiny scale via --set overrides, so it's
# a real check of the mode-collapse fix, not just pipeline plumbing.
# gradient_accumulation_steps=1 (not the config's 16): with only 20 samples
# / batch_size=2 = 10 batches, grad_accum=16 would never complete even one
# accumulation window, so zero optimizer steps would fire and there would
# be no loss to inspect.
# NOTE: --set uses argparse nargs="*" — pass ALL overrides in ONE --set
# flag. Repeating --set multiple times makes each occurrence REPLACE the
# previous one's value rather than accumulate (confirmed the hard way: an
# earlier version of this script with six separate --set flags silently
# applied only the last one and ran a full 5-epoch/47149-sample job).
python scripts/train_ctclip.py \
  --config configs/ctclip_stage2.yaml \
  --set max_samples=20 num_epochs=1 gradient_accumulation_steps=1 \
        val_every_n_steps=5 save_every_n_steps=10 patience=99 \
        checkpoint_dir=/scratch/sadeniji/smoke_checkpoints_biomedlm
