#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --job-name=ictc_eval_stage2_par
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=200G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/sadeniji/ictc/logs/eval_stage2_par_%j.out
#SBATCH --error=/scratch/sadeniji/ictc/logs/eval_stage2_par_%j.err

# ── Environment ──────────────────────────────────────────────────────────
# Same setup as scripts/slurm_eval.sh. Plain independent `python3` launches,
# not srun/deepspeed — each shard is a fully separate process with no
# torch.distributed group, no collectives, nothing to keep in sync across
# shards (every sample's generation is independent), so there's no reason to
# use a distributed launcher here. Backgrounding + `wait` is simpler and
# avoids depending on SLURM's own multi-task GPU-binding behavior, which
# this project has never verified on this cluster.
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

NUM_SHARDS=4
OUTPUT=/scratch/sadeniji/ictc/eval/stage2_e2e_eval_n3039.csv

for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    python3 scripts/qualitative_eval_v3.py \
        --checkpoint-dir /scratch/sadeniji/ictc_checkpoints_e2e_stage2/checkpoint_best \
        --config configs/ctclip_e2e_stage2.yaml \
        --num-samples 3039 \
        --output "$OUTPUT" \
        --shard-id $shard \
        --num-shards $NUM_SHARDS \
        > /scratch/sadeniji/ictc/logs/eval_stage2_par_${SLURM_JOB_ID}_shard${shard}.out \
        2> /scratch/sadeniji/ictc/logs/eval_stage2_par_${SLURM_JOB_ID}_shard${shard}.err &
done

echo "Launched $NUM_SHARDS shard processes, waiting..."
wait
echo "All shards finished at $(date)"

python3 scripts/merge_eval_shards.py --output "$OUTPUT" --num-shards $NUM_SHARDS

echo "Eval done at $(date)"
