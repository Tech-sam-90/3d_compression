#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=ictc_qual_v2_t2t3
#SBATCH --output=/scratch/sadeniji/logs/qual_eval_v2_t2t3_%j.out
#SBATCH --error=/scratch/sadeniji/logs/qual_eval_v2_t2t3_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

python scripts/qualitative_eval_v2.py \
  --ckpt /scratch/sadeniji/ictc_checkpoints_t2t3_fix/checkpoint_best.pt \
  --config configs/ctclip_t2t3_fix.yaml \
  --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid \
  --out docs/QUALITATIVE_EVAL_T2T3_FIX.md
