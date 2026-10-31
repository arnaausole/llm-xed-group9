#!/bin/bash
# Evaluate an already trained QLoRA adapter on Hábrók (no training). Submit from the repository root:
#   sbatch jobs/peft_eval.slurm configs/peft/qwen3-8b_qlora_en.yaml [dev|test]
#SBATCH --job-name=xed-peft-eval
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch jobs/peft_eval.slurm <config.yaml> [dev|test]}
SPLIT=${2:-test}

# Keep model downloads and caches out of the 50 GB home directory.
export HF_HOME=/scratch/$USER/hf_cache
export UV_CACHE_DIR=/scratch/$USER/uv_cache

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
uv run python peft/evaluate_peft.py --config "$CONFIG" --split "$SPLIT"
