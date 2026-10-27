#!/bin/bash
#SBATCH --account=rpp-uanazodo
#SBATCH --job-name=ictc_stage1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=100G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/sadeniji/ictc/logs/stage1_%j.out
#SBATCH --error=/scratch/sadeniji/ictc/logs/stage1_%j.err

# ── Environment ──────────────────────────────────────────────────────────
module purge
# StdEnv/2023 gcc/12.3 cuda/13.2: torch here is built against CUDA 13.2
# (verified: python -c "import torch; print(torch.version.cuda)" -> 13.2).
# DeepSpeed's op-builder requires an exact CUDA major.minor match against
# torch's build or refuses to JIT-compile DeepSpeedCPUAdam
# (CUDAMismatchException) — confirmed the hard way in the Prompt 4 dry-run
# debugging session; do not swap this for a different cuda/ version without
# re-checking torch.version.cuda first. gcc/12.3 is cuda/13.2's required
# prerequisite (module spider cuda/13.2). arrow/25.0.0 provides pyarrow —
# do NOT let pip try to install it, it resolves to a deliberate dummy
# blocker package on this cluster.
module load StdEnv/2023 gcc/12.3 cuda/13.2 arrow/25.0.0

source ~/envs/ictc/bin/activate

# NOT /scratch/sadeniji/ct_clip_src on PYTHONPATH: that outer repo dir
# contains a transformer_maskgit/ subfolder that is NOT the importable
# package (the real one is one level deeper, transformer_maskgit/
# transformer_maskgit/), so Python treats the outer dir as an empty
# implicit namespace package and shadows the real pip-editable-installed
# package (`pip install --no-index --no-deps -e
# /scratch/sadeniji/ct_clip_src/transformer_maskgit`, already registered in
# site-packages) — causes ImportError: cannot import name 'CTViT' from
# 'transformer_maskgit' (unknown location). Confirmed the hard way: this
# broke the first real stage1 submission. The editable install alone is
# sufficient; only the pyarrow site-packages path needs to be added here.
export PYTHONPATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/arrow/25.0.0/lib/python3.11/site-packages:$PYTHONPATH

# Compute nodes have no internet access. Without HF_HUB_OFFLINE/
# TRANSFORMERS_OFFLINE, transformers' from_pretrained() still attempts an
# HTTPS metadata check and hangs indefinitely (TCP SYN never gets a
# response) instead of using the cache — see scripts/train_ictc_llama3b.sh's
# own comment for this exact, previously-hit failure mode in this project.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/project/def-uanazodo-ab/sadeniji/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# CUDA_HOME/PATH/LD_LIBRARY_PATH are already set correctly by the
# `module load cuda/13.2` above — no manual export needed (and a
# hand-typed path risks silently drifting out of sync with whichever
# cuda/ module is actually loaded).

# DeepSpeed communication
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1   # InfiniBand caused hangs in prior Narval sessions on this project — do not re-enable
export OMP_NUM_THREADS=4

# DeepSpeed's process-group collective timeout (NOT an NCCL-level env var —
# NCCL_TIMEOUT does not control this; DeepSpeed reads DEEPSPEED_TIMEOUT
# specifically, in MINUTES: deepspeed/constants.py's
# `default_pg_timeout = timedelta(minutes=int(os.getenv("DEEPSPEED_TIMEOUT", default=30)))`.
# Raised from the 30-minute default to give a slow-but-genuinely-progressing
# checkpoint write more room before the watchdog kills the whole job.
export DEEPSPEED_TIMEOUT=120   # 120 minutes = 2 hours

# ── Directories ──────────────────────────────────────────────────────────
mkdir -p /scratch/sadeniji/ictc/logs
mkdir -p /scratch/sadeniji/ictc_checkpoints_e2e_stage1

# ── Launch ───────────────────────────────────────────────────────────────
cd ~/3d_compression

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

deepspeed --num_gpus=4 \
    scripts/train_ctclip.py \
    --config configs/ctclip_e2e_stage1.yaml

echo "Stage1 done at $(date)"
