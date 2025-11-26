#!/bin/bash
#SBATCH --job-name=lora_array
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100-16:1
#SBATCH --mem=20G
#SBATCH --time=24:00:00
#SBATCH --array=0-35
#SBATCH --output=/ocean/projects/cis250187p/dgupta4/speech-adapter-routing/logs/array_lora_%A_%a.out
#SBATCH --error=/ocean/projects/cis250187p/dgupta4/speech-adapter-routing/logs/array_lora_%A_%a.err
# #SBATCH --mail-type=END,FAIL

# ============================================
# LoRA Training Job Array
# Trains adapters for all model-language-rank combinations
# Total: 3 models × 4 languages × 3 ranks = 36 jobs
# ============================================

# Define experiment grid
MODELS=("whisper-small" "whisper-medium" "whisper-large")
LANGUAGES=("hindi" "italian" "punjabi" "telugu")
RANKS=(8 16 32)

# Calculate total combinations
NUM_MODELS=${#MODELS[@]}
NUM_LANGUAGES=${#LANGUAGES[@]}
NUM_RANKS=${#RANKS[@]}

# Compute indices from array task ID
# Layout: (model_idx * NUM_LANGUAGES * NUM_RANKS) + (lang_idx * NUM_RANKS) + rank_idx
MODEL_IDX=$((SLURM_ARRAY_TASK_ID / (NUM_LANGUAGES * NUM_RANKS)))
REMAINDER=$((SLURM_ARRAY_TASK_ID % (NUM_LANGUAGES * NUM_RANKS)))
LANG_IDX=$((REMAINDER / NUM_RANKS))
RANK_IDX=$((REMAINDER % NUM_RANKS))

MODEL=${MODELS[$MODEL_IDX]}
LANGUAGE=${LANGUAGES[$LANG_IDX]}
RANK=${RANKS[$RANK_IDX]}

# Set data sources based on language
# Italian: Common Voice + MLS
# Indic languages (Hindi, Punjabi, Telugu): Common Voice + AI4Bharat
if [ "$LANGUAGE" == "italian" ]; then
    DATA_SOURCES="common_voice mls"
else
    DATA_SOURCES="common_voice ai4bharat"
fi

echo "============================================"
echo "SLURM Job Array Task: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Language: ${LANGUAGE}"
echo "LoRA Rank: ${RANK}"
echo "Data Sources: ${DATA_SOURCES}"
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
    --data_sources $DATA_SOURCES \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 5e-4 \
    --warmup_steps 500 \
    --max_steps 5000 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --mixed_precision bf16 \
    --output_dir $OUTPUT_DIR \
    --wandb_project whisper-lora-adapters \
    --wandb_run_name ${MODEL}_${LANGUAGE}_r${RANK} \
    --seed 42

echo "Training completed for ${MODEL} ${LANGUAGE} r${RANK}"
