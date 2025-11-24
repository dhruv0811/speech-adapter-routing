#!/bin/bash
# ============================================
# UV Environment Setup Script
# ============================================

set -e

echo "Setting up UV environment for speech-adapter-routing..."

# Set project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $PROJECT_DIR

# Set cache directories
export UV_CACHE_DIR=${PROJECT:-$HOME}/.cache/uv
export HF_HOME=${PROJECT:-$HOME}/.cache/huggingface
export TRANSFORMERS_CACHE=${PROJECT:-$HOME}/.cache/transformers

mkdir -p $UV_CACHE_DIR
mkdir -p $HF_HOME
mkdir -p $TRANSFORMERS_CACHE

echo "Cache directories:"
echo "  UV_CACHE_DIR: $UV_CACHE_DIR"
echo "  HF_HOME: $HF_HOME"

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "UV not found. Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "UV version: $(uv --version)"

# Sync dependencies
echo ""
echo "Syncing dependencies..."
uv sync

# Verify installation
echo ""
echo "Verifying installation..."
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
uv run python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
uv run python -c "import peft; print(f'PEFT: {peft.__version__}')"

echo ""
echo "============================================"
echo "Setup complete!"
echo ""
echo "To run commands, use: uv run <command>"
echo "Example: uv run python scripts/train_lora.py --help"
echo "============================================"
