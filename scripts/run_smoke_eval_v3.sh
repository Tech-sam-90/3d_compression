#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --job-name=ictc_eval_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/sadeniji/ictc/logs/eval_smoke_%j.out
#SBATCH --error=/scratch/sadeniji/ictc/logs/eval_smoke_%j.err

module purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/25.0.0
source ~/envs/ictc/bin/activate

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

mkdir -p /scratch/sadeniji/ictc/logs

cd ~/3d_compression
python3 scripts/qualitative_eval_v3.py \
    --checkpoint-dir /scratch/sadeniji/ictc_checkpoints_vfilm_fix \
    --config configs/ctclip_e2e_stage1.yaml \
    --num-samples 5 \
    --output /scratch/sadeniji/ictc/eval/smoke_test.csv
