#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --job-name=ictc_eval_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/sadeniji/ictc/logs/eval_stage2_%j.out
#SBATCH --error=/scratch/sadeniji/ictc/logs/eval_stage2_%j.err

# ── Environment ──────────────────────────────────────────────────────────
# Same setup as scripts/slurm_eval.sh — see that file's comments for why
# each piece is needed (CUDA 13.2 match, PYTHONPATH namespace-package trap,
# HF offline mode). This script is plain single-GPU inference, not a
# DeepSpeed launch, so no deepspeed launcher / NCCL env vars are needed.
module purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/25.0.0

source ~/envs/ictc/bin/activate

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# ── Directories ──────────────────────────────────────────────────────────
mkdir -p /scratch/sadeniji/ictc/logs
mkdir -p /scratch/sadeniji/ictc/eval

# ── Launch ───────────────────────────────────────────────────────────────
cd ~/3d_compression

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

# --checkpoint-dir points at checkpoint_best/ specifically (not the parent
# ictc_checkpoints_e2e_stage2/ dir) — stage2's training loop only writes the
# flat, directly-loadable pytorch_model.bin inside checkpoint_best/ (via
# save_16bit_model on val improvement); unlike stage1, it never writes a
# top-level checkpoint_latest_consolidated.pt (that save is gated to
# training_stage == "stage1" in scripts/train_ctclip.py, since only stage1
# needs to hand off a warm-start file to a next stage).
python3 scripts/qualitative_eval_v3.py \
    --checkpoint-dir /scratch/sadeniji/ictc_checkpoints_e2e_stage2/checkpoint_best \
    --config configs/ctclip_e2e_stage2.yaml \
    --num-samples 3039 \
    --output /scratch/sadeniji/ictc/eval/stage2_e2e_eval_n3039.csv

echo "Eval done at $(date)"
