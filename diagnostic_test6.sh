#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --job-name=ictc_diag_test6
#SBATCH --output=/scratch/sadeniji/logs/diag_test6_%j.out
#SBATCH --error=/scratch/sadeniji/logs/diag_test6_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

echo "=== Test 6 vs joint single-stage baseline checkpoint ==="
python3 scratch_diagnostic_test6.py \
  /scratch/sadeniji/ictc_checkpoints/checkpoint_best.pt \
  joint_baseline \
  configs/ctclip_stage2.yaml

echo ""
echo "=== Test 6 vs two-stage final checkpoint (this session's result) ==="
python3 scratch_diagnostic_test6.py \
  /scratch/sadeniji/ictc_checkpoints_stage2_final/checkpoint_best.pt \
  two_stage_final \
  configs/ctclip_stage2_final.yaml
