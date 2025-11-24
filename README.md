# Multilingual ASR Adapters with LoRA

This project explores adapter-based parameter-efficient fine-tuning for multilingual ASR using LoRA adapters on Whisper models.

## Project Structure

```
speech-adapter-routing/
├── configs/                    # Configuration files
│   ├── model_configs/          # Model hyperparameters
│   ├── lora_configs/           # LoRA settings
│   ├── training_configs/       # Training hyperparams
│   └── dataset_configs/        # Data loading configs
├── src/                        # Source code
│   ├── data/                   # Data loading & preprocessing
│   ├── models/                 # Model wrappers (WhisperLoRA, router)
│   ├── training/               # Training utilities
│   └── evaluation/             # Evaluation utilities
├── scripts/                    # Main scripts
│   ├── train_lora.py          # Training script
│   └── evaluate_model.py      # Evaluation script
├── slurm_jobs/                # SLURM job scripts
│   ├── train_lora_array.sh    # Job array for training
│   ├── train_single.sh        # Single training job
│   └── evaluate.sh            # Evaluation job
├── checkpoints/               # Saved model checkpoints
├── results/                   # Evaluation results
└── logs/                      # Training/job logs
```

## Setup with UV

This project uses [UV](https://github.com/astral-sh/uv) for fast, reliable Python package management.

### 1. Install UV (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set Environment Variables

```bash
export PROJECT=/ocean/projects/cis250187p/dgupta4
export UV_CACHE_DIR=$PROJECT/.cache/uv
export HF_HOME=$PROJECT/.cache/huggingface
export TRANSFORMERS_CACHE=$PROJECT/.cache/transformers
```

### 3. Setup Environment

```bash
cd $PROJECT/speech-adapter-routing
bash setup_uv.sh
```

Or manually:

```bash
uv sync
```

## Quick Start

### Train a Single Adapter

```bash
# Interactive (for debugging)
uv run python scripts/train_lora.py \
    --model whisper-small \
    --language hindi \
    --lora_rank 16 \
    --max_steps 100 \
    --output_dir checkpoints/test \
    --no_wandb

# SLURM job
sbatch slurm_jobs/train_single.sh whisper-small hindi 16
```

### Submit All Training Jobs

```bash
# Submit job array (20 jobs: 2 models × 4 languages × 2 ranks)
bash slurm_jobs/submit_all_phase1.sh
```

### Evaluate a Trained Adapter

```bash
uv run python scripts/evaluate_model.py \
    --model whisper-small \
    --checkpoint checkpoints/whisper-small_hindi_r16/best \
    --language hindi \
    --output_dir results/whisper-small_hindi
```

## Languages Supported

| Language | Code | Dataset Sources |
|----------|------|-----------------|
| Hindi    | hi   | Common Voice, AI4Bharat |
| Italian  | it   | Common Voice, MLS |
| Punjabi  | pa   | Common Voice, AI4Bharat |
| Telugu   | te   | Common Voice, AI4Bharat |

## Models Supported

| Model | Parameters | Hidden Dim | Layers |
|-------|------------|------------|--------|
| whisper-small | 244M | 768 | 12 |
| whisper-medium | 769M | 1024 | 24 |
| whisper-large | 1.55B | 1280 | 32 |

## LoRA Configuration

Default settings:
- **Rank (r)**: 16 (test 8, 16, 32, 64)
- **Alpha**: 32 (2 × rank)
- **Dropout**: 0.1
- **Target modules**: `q_proj`, `v_proj`

## Training Configuration

Default hyperparameters:
- **Batch size**: 16 (effective 64 with accumulation)
- **Learning rate**: 5e-4
- **Warmup steps**: 500
- **Max steps**: 5000
- **Mixed precision**: bf16

## Project Phases

### Phase 1: LoRA Adapter Training (Current)
Train language-specific adapters for Whisper models.

### Phase 2: Adaptive Routing
Build language detection + adapter routing system.

### Phase 3: Analysis & Interpretability
Analyze adapter weights, transfer learning, linguistic patterns.

## Monitoring

```bash
# Check job status
squeue -u $USER

# Monitor training logs
tail -f logs/lora_*.out

# Check results
bash slurm_jobs/monitor_jobs.sh
```

## Running with UV

All Python commands should be prefixed with `uv run`:

```bash
# Run training
uv run python scripts/train_lora.py --help

# Run evaluation
uv run python scripts/evaluate_model.py --help

# Run tests
uv run python test_setup.py

# Interactive Python
uv run python
```

## Citation

If you use this code, please cite:

```bibtex
@misc{multilingual-asr-adapters,
  title={Multilingual ASR Adapters with LoRA},
  author={Gupta, Dhruv and Vigano, Andrea and Kalra, Jushaan and Thammineni, Swaroop and Bharadwaj, Shikhar},
  year={2024},
  institution={Carnegie Mellon University}
}
```

## License

MIT License
