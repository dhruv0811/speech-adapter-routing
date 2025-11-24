#!/bin/bash
#SBATCH --job-name=eval_lora
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=22G
#SBATCH --time=4:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# ============================================
# Evaluation Job for trained adapters
# Usage: sbatch evaluate.sh <checkpoint_path> <model> <language>
# Example: sbatch evaluate.sh checkpoints/whisper-small_hindi_r16/best whisper-small hindi
# ============================================

CHECKPOINT=${1:-"checkpoints/whisper-small_hindi_r16/best"}
MODEL=${2:-"whisper-small"}
LANGUAGE=${3:-"hindi"}

echo "============================================"
echo "Evaluation Configuration"
echo "Checkpoint: ${CHECKPOINT}"
echo "Model: ${MODEL}"
echo "Language: ${LANGUAGE}"
echo "============================================"

# Set up environment
cd $PROJECT/speech-adapter-routing

# Set cache directories
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/transformers
export UV_CACHE_DIR=$PROJECT/.cache/uv

# Create output directory
OUTPUT_DIR="results/${MODEL}_${LANGUAGE}"
mkdir -p $OUTPUT_DIR

# Run evaluation using UV
uv run python scripts/evaluate_model.py \
    --model $MODEL \
    --checkpoint $CHECKPOINT \
    --language $LANGUAGE \
    --data_sources common_voice \
    --split test \
    --batch_size 16 \
    --num_beams 1 \
    --output_dir $OUTPUT_DIR \
    --save_predictions

echo "Evaluation completed!"
