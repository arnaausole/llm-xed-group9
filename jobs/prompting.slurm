#!/bin/bash
# Prompting run on Hábrók. Submit from the repository root:
#   sbatch jobs/prompting.slurm configs/prompting/qwen3-8b_zero-shot.yaml [dev|test] [extra args]
# e.g. a smoke test:  sbatch --time=00:30:00 jobs/prompting.slurm <config> dev --limit 50
#SBATCH --job-name=xed-prompting
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch jobs/prompting.slurm <config.yaml> [dev|test] [extra args]}
SPLIT=${2:-dev}
shift $(( $# >= 2 ? 2 : 1 ))

# Keep model downloads and caches out of the 50 GB home directory.
export HF_HOME=/scratch/$USER/hf_cache
export UV_CACHE_DIR=/scratch/$USER/uv_cache

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
uv run python prompting/run_prompting.py --config "$CONFIG" --split "$SPLIT" "$@"
