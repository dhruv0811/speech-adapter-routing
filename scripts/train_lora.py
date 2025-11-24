"""Main training script for LoRA adapter fine-tuning."""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import WhisperLoRA, get_processor
from src.data import create_dataset, DataCollatorSpeechSeq2Seq
from src.training import (
    ASRTrainer,
    WandbCallback,
    CheckpointCallback,
    EarlyStoppingCallback,
)

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LoRA adapters for ASR")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        choices=["whisper-small", "whisper-medium", "whisper-large"],
        help="Base model to use",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["hindi", "italian", "punjabi", "telugu"],
        help="Target language",
    )
    
    # LoRA arguments
    parser.add_argument("--lora_rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=["q_proj", "v_proj"],
        help="Target modules for LoRA",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_sources",
        type=str,
        nargs="+",
        default=["common_voice"],
        help="Data sources to use",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum training samples")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Maximum audio duration")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Minimum audio duration")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=5000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation interval")
    parser.add_argument("--scheduler_type", type=str, default="linear", help="LR scheduler type")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training",
    )
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--save_steps", type=int, default=1000, help="Checkpoint save interval")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum checkpoints to keep")
    
    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="whisper-lora-adapters", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B logging")
    
    # Other arguments
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from checkpoint")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    """Main training function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args)
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    logger.info(f"Training configuration:\n{yaml.dump(config, default_flow_style=False)}")
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = WhisperLoRA(
        model_name=args.model,
        lora_r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        language=args.language,
        device=device,
        use_gradient_checkpointing=True,
    )
    
    processor = model.processor
    
    # Load datasets
    logger.info(f"Loading datasets for {args.language}")
    
    train_dataset = create_dataset(
        language=args.language,
        split="train",
        sources=args.data_sources,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        processor=processor,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )
    
    eval_dataset = create_dataset(
        language=args.language,
        split="validation",
        sources=args.data_sources,
        max_samples=args.max_samples // 10 if args.max_samples else None,
        cache_dir=args.cache_dir,
        processor=processor,
        max_duration=args.max_duration,
        min_duration=args.min_duration,
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")
    
    # Create data collator
    data_collator = DataCollatorSpeechSeq2Seq(processor=processor)
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )
    
    # Setup callbacks
    callbacks = []
    
    # W&B callback
    if not args.no_wandb:
        wandb_run_name = args.wandb_run_name or f"{args.model}_{args.language}_r{args.lora_rank}"
        callbacks.append(WandbCallback(
            project=args.wandb_project,
            name=wandb_run_name,
            config=config,
        ))
    
    # Checkpoint callback
    callbacks.append(CheckpointCallback(
        save_dir=output_dir / "checkpoints",
        save_every=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_best_only=True,
        metric_name="wer",
        greater_is_better=False,
    ))
    
    # Early stopping callback
    callbacks.append(EarlyStoppingCallback(
        patience=args.early_stopping_patience,
        metric_name="wer",
        greater_is_better=False,
    ))
    
    # Create trainer
    trainer = ASRTrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        processor=processor,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scheduler_type=args.scheduler_type,
        mixed_precision=args.mixed_precision,
        callbacks=callbacks,
        device=device,
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        logger.info(f"Resuming from checkpoint: {args.resume_from}")
        trainer.load_checkpoint(args.resume_from)
    
    # Train
    logger.info("Starting training...")
    final_metrics = trainer.train()
    
    # Save final model
    final_path = output_dir / "final"
    model.save_adapter(final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Log final metrics
    logger.info(f"Training completed. Final metrics: {final_metrics}")
    
    return final_metrics


if __name__ == "__main__":
    main()
