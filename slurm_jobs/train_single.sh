#!/bin/bash
#SBATCH --job-name=lora_single
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-16:1
#SBATCH --mem=20G
#SBATCH --time=12:00:00
#SBATCH --output=/ocean/projects/cis250187p/dgupta4/speech-adapter-routing/logs/lora_%j.out
#SBATCH --error=/ocean/projects/cis250187p/dgupta4/speech-adapter-routing/logs/lora_%j.err
# #SBATCH --mail-type=END,FAIL

# ============================================
# Single LoRA Training Job
# Usage: sbatch train_single.sh <model> <language> <rank>
# Example: sbatch train_single.sh whisper-small hindi 16
# ============================================

# Parse arguments
MODEL=${1:-"whisper-small"}
LANGUAGE=${2:-"hindi"}
RANK=${3:-16}

echo "============================================"
echo "Training Configuration"
echo "Model: ${MODEL}"
echo "Language: ${LANGUAGE}"
echo "LoRA Rank: ${RANK}"
echo "============================================"

# Set up environment
cd $PROJECT/speech-adapter-routing

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    export $(grep -v '^#' .env | xargs)
fi

# Set cache directories
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/transformers
export WANDB_DIR=$PROJECT/speech-adapter-routing/logs
export UV_CACHE_DIR=$PROJECT/.cache/uv

# Create output directory
OUTPUT_DIR="checkpoints/${MODEL}_${LANGUAGE}_r${RANK}"
mkdir -p $OUTPUT_DIR
mkdir -p logs

# Run training using UV
uv run python scripts/train_lora.py \
    --model $MODEL \
    --language $LANGUAGE \
    --lora_rank $RANK \
    --lora_alpha $((RANK * 2)) \
    --lora_dropout 0.1 \
    --data_sources common_voice \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --warmup_steps 500 \
    --max_steps 5000 \
    --eval_steps 500 \
    --save_steps 1000 \
    --mixed_precision bf16 \
    --output_dir $OUTPUT_DIR \
    --wandb_project whisper-lora-adapters \
    --wandb_run_name ${MODEL}_${LANGUAGE}_r${RANK} \
    --seed 42

echo "Training completed!"
