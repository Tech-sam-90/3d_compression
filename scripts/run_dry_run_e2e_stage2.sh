#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=ictc_e2e_dryrun_s2
#SBATCH --output=/scratch/sadeniji/logs/e2e_dryrun_s2_%j.out
#SBATCH --error=/scratch/sadeniji/logs/e2e_dryrun_s2_%j.err

mkdir -p /scratch/sadeniji/logs

module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/25.0.0
source ~/envs/ictc/bin/activate

export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export MASTER_PORT=29501

cd ~/3d_compression
deepspeed --num_gpus=1 scripts/train_ctclip.py \
    --config configs/ctclip_e2e_stage2.yaml \
    --max-steps 3 \
    --dry-run
