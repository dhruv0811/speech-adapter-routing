#!/bin/bash
# ============================================
# Submit all Phase 1 training jobs
# ============================================

echo "Submitting Phase 1 LoRA training jobs..."

# Create logs directory
mkdir -p logs

# Method 1: Submit job array (recommended)
echo "Submitting job array..."
JOB_ID=$(sbatch slurm_jobs/train_lora_array.sh | awk '{print $4}')
echo "Submitted job array with ID: $JOB_ID"

# Method 2: Submit individual jobs (alternative)
# Uncomment below to use individual jobs instead of array

# MODELS=("whisper-small" "whisper-medium")
# LANGUAGES=("hindi" "italian" "punjabi" "telugu")
# RANKS=(16 32)
#
# for MODEL in "${MODELS[@]}"; do
#     for LANGUAGE in "${LANGUAGES[@]}"; do
#         for RANK in "${RANKS[@]}"; do
#             echo "Submitting: $MODEL $LANGUAGE r$RANK"
#             sbatch slurm_jobs/train_single.sh $MODEL $LANGUAGE $RANK
#             sleep 1  # Small delay between submissions
#         done
#     done
# done

echo "All jobs submitted!"
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Cancel all with: scancel -u \$USER"
