#!/bin/bash
#SBATCH --job-name=lid_train
#SBATCH --partition=GPU-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100-32:1
#SBATCH --mem=64GB
#SBATCH --time=8:00:00
#SBATCH --output=/ocean/projects/cis250187p/dgupta4/speech-adapter-routing/logs/lid_%j.out
#SBATCH --error=/ocean/projects/cis250187p/dgupta4/speech-adapter-routing/logs/lid_%j.err

# ============================================================================
# SLURM Job Script for LID Classifier Training
# ============================================================================

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Project paths
PROJECT=/ocean/projects/cis250187p/dgupta4
WORKDIR=$PROJECT/speech-adapter-routing

cd $WORKDIR

# Setup environment
echo "Setting up environment..."
source $PROJECT/.venv/bin/activate 2>/dev/null || source ~/.bashrc

# If using UV
if [ -f "$WORKDIR/.venv/bin/activate" ]; then
    source $WORKDIR/.venv/bin/activate
fi

# Environment variables
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/huggingface
export HF_DATASETS_CACHE=$PROJECT/.cache/huggingface/datasets
export WANDB_DIR=$WORKDIR/logs
export WANDB_CACHE_DIR=$PROJECT/.cache/wandb

# Configuration
BASE_MODEL=${BASE_MODEL:-"whisper-small"}
LANGUAGES=${LANGUAGES:-"hindi italian punjabi telugu"}
SAMPLES_PER_LANG=${SAMPLES_PER_LANG:-5000}
MAX_STEPS=${MAX_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-32}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
POOLING=${POOLING:-"mean"}
USE_CNN=${USE_CNN:-""}

# Output directory
OUTPUT_DIR=$WORKDIR/checkpoints/lid_classifier_${BASE_MODEL}

# Build command
CMD="python scripts/train_router.py \
    --base_model $BASE_MODEL \
    --languages $LANGUAGES \
    --data_sources common_voice \
    --samples_per_language $SAMPLES_PER_LANG \
    --max_steps $MAX_STEPS \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --pooling $POOLING \
    --warmup_steps 100 \
    --eval_steps 200 \
    --mixed_precision bf16 \
    --output_dir $OUTPUT_DIR \
    --wandb_project lid-classifier \
    --wandb_run_name lid_${BASE_MODEL} \
    --cache_dir $HF_DATASETS_CACHE \
    --balanced"

# Add CNN flag if specified
if [ -n "$USE_CNN" ]; then
    CMD="$CMD --use_cnn"
fi

echo "=========================================="
echo "Running command:"
echo "$CMD"
echo "=========================================="

# Run training
$CMD

EXIT_CODE=$?

echo "=========================================="
echo "Job finished with exit code: $EXIT_CODE"
echo "End Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
