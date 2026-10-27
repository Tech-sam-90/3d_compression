#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --job-name=ictc_t1_metrics
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=40G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/sadeniji/ictc/logs/t1_metrics_%j.out
#SBATCH --error=/scratch/sadeniji/ictc/logs/t1_metrics_%j.err

# No generation here — just re-scoring existing qualitative_eval_v3.py CSVs
# (already on disk) with compute_all_metrics()'s full BLEU-4/METEOR/ROUGE-L/
# CIDEr/GREEN/RaTEScore/RadGraph-XL-F1 suite, so this should be much faster
# than the eval jobs that produced the CSVs.
module purge
module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/25.0.0

source ~/envs/ictc/bin/activate

export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

cd ~/3d_compression

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

echo "=== Stage1 ==="
python3 scripts/compute_full_t1_metrics.py \
    --csv /scratch/sadeniji/ictc/eval/stage1_e2e_eval_n3039.csv \
    --output /scratch/sadeniji/ictc/eval/stage1_e2e_t1_full_metrics.json

echo
echo "=== Stage2 ==="
python3 scripts/compute_full_t1_metrics.py \
    --csv /scratch/sadeniji/ictc/eval/stage2_e2e_eval_n3039.csv \
    --output /scratch/sadeniji/ictc/eval/stage2_e2e_t1_full_metrics.json

echo "Done at $(date)"
