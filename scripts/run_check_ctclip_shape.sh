#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --job-name=ictc_ctclip_shape
#SBATCH --output=/scratch/sadeniji/logs/ctclip_shape_%j.out
#SBATCH --error=/scratch/sadeniji/logs/ctclip_shape_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export PYTHONPATH="/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:/scratch/sadeniji/ct_clip_src/transformer_maskgit:$PYTHONPATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression
python3 scripts/check_ctclip_shape.py
