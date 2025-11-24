#!/bin/bash
# ============================================
# Utility script to monitor jobs and aggregate results
# ============================================

# Check job status
echo "Current job status:"
squeue -u $USER

echo ""
echo "============================================"
echo "Completed jobs:"

# Find completed checkpoints
echo ""
echo "Available checkpoints:"
find checkpoints -name "best" -type d 2>/dev/null

echo ""
echo "============================================"
echo "Training logs with errors:"
grep -l "Error\|Exception\|CUDA" logs/lora_*.err 2>/dev/null | head -10

echo ""
echo "============================================"
echo "Best WER results:"
cd $PROJECT/speech-adapter-routing
for metrics_file in results/*/metrics.json; do
    if [ -f "$metrics_file" ]; then
        echo "$(dirname $metrics_file): $(cat $metrics_file | uv run python -c "import sys, json; d=json.load(sys.stdin); print(f\"WER={d['wer']:.4f}\")")"
    fi
done 2>/dev/null

echo ""
echo "============================================"
echo "GPU utilization (if jobs running):"
nvidia-smi --query-gpu=gpu_name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo "No GPU info available"
