#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --job-name=ictc_qualitative_eval
#SBATCH --output=/scratch/sadeniji/logs/qualitative_eval_%j.out
#SBATCH --error=/scratch/sadeniji/logs/qualitative_eval_%j.err

mkdir -p /scratch/sadeniji/logs

module load gcc arrow/25.0.0
source ~/envs/ictc/bin/activate

# Compute nodes have no internet access — without HF_HUB_OFFLINE/
# TRANSFORMERS_OFFLINE, transformers' from_pretrained() still attempts an
# HTTPS metadata check and hangs indefinitely instead of using the cache
# (same reasoning as every other launcher script in scripts/).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache
export HF_DATASETS_CACHE=/project/def-uanazodo-ab/sadeniji/hf_cache

cd ~/3d_compression

if [ -z "$ICTC_CKPT" ]; then
  echo "ERROR: set ICTC_CKPT to the checkpoint path, e.g.:" >&2
  echo "  ICTC_CKPT=/path/to/checkpoint_best.pt sbatch scripts/qualitative_eval.sh" >&2
  exit 1
fi

CONFIG="${ICTC_CONFIG:-configs/ctclip_stage2_final.yaml}"
REPORT_CSV="${ICTC_REPORT_CSV:-/project/def-uanazodo-ab/sadeniji/ctrate_csv/dataset/merged/valid_merged.csv}"

python scripts/qualitative_eval.py \
  --ckpt "$ICTC_CKPT" \
  --config "$CONFIG" \
  --feature_dir /project/def-uanazodo-ab/sadeniji/ctrate_features/valid \
  --report_csv "$REPORT_CSV"
