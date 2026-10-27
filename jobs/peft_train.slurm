#!/bin/bash
# QLoRA training + development-set evaluation on Hábrók. Submit from the repository root:
#   sbatch jobs/peft_train.slurm configs/peft/qwen3-8b_qlora_en.yaml
# The bilingual condition has twice as many examples: add --time=10:00:00.
#SBATCH --job-name=xed-qlora
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail
CONFIG=${1:?usage: sbatch jobs/peft_train.slurm <config.yaml>}

# Keep model downloads and caches out of the 50 GB home directory.
export HF_HOME=/scratch/$USER/hf_cache
export UV_CACHE_DIR=/scratch/$USER/uv_cache

cd "$SLURM_SUBMIT_DIR"
nvidia-smi
uv run python peft/train_qlora.py --config "$CONFIG"
uv run python peft/evaluate_peft.py --config "$CONFIG" --split dev
