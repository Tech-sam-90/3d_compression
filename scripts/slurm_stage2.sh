#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --job-name=ictc_stage2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=220G
#SBATCH --time=3-00:00:00
#SBATCH --output=/scratch/sadeniji/ictc/logs/stage2_%j.out
#SBATCH --error=/scratch/sadeniji/ictc/logs/stage2_%j.err

# ── Environment ──────────────────────────────────────────────────────────
module purge
# See scripts/slurm_stage1.sh's comment — same CUDA 13.2 / DeepSpeedCPUAdam
# constraint applies here unchanged.
module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/25.0.0

source ~/envs/ictc/bin/activate

# See scripts/slurm_stage1.sh's comment — /scratch/sadeniji/ct_clip_src
# must NOT be on PYTHONPATH, it shadows the real editable-installed
# transformer_maskgit package with a broken empty namespace package.
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1   # InfiniBand caused hangs in prior Narval sessions on this project — do not re-enable
export OMP_NUM_THREADS=4

# ── Directories ──────────────────────────────────────────────────────────
mkdir -p /scratch/sadeniji/ictc/logs
mkdir -p /scratch/sadeniji/ictc_checkpoints_e2e_stage2

# ── Launch ───────────────────────────────────────────────────────────────
cd ~/3d_compression

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

deepspeed --num_gpus=4 \
    scripts/train_ctclip.py \
    --config configs/ctclip_e2e_stage2.yaml

echo "Stage2 done at $(date)"
