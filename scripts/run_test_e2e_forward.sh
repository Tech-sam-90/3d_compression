#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --job-name=ictc_e2e_fwd
#SBATCH --output=/scratch/sadeniji/logs/e2e_fwd_%j.out
#SBATCH --error=/scratch/sadeniji/logs/e2e_fwd_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression
python3 scripts/test_e2e_forward.py
