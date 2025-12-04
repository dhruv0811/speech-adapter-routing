"""Training script for Language Identification (LID) classifier for adapter routing."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)
import yaml
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.adapter_router import LanguageClassifier, EncoderFeatureExtractor
from src.models.base import load_base_model, get_processor
from src.data import create_dataset

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset for LID Training
# ============================================================================

class LIDDataset(Dataset):
    """Dataset for Language Identification training.
    
    Wraps audio samples with language labels for classifier training.
    """
    
    def __init__(
        self,
        datasets_by_language: Dict[str, Dataset],
        languages: List[str],
        samples_per_language: Optional[int] = None,
        balanced: bool = True,
    ):
        """
        Args:
            datasets_by_language: Dict mapping language name to ASR dataset
            languages: Ordered list of language names (defines class indices)
            samples_per_language: Max samples per language (for balancing)
            balanced: Whether to balance classes by undersampling
        """
        self.languages = languages
        self.lang_to_idx = {lang: i for i, lang in enumerate(languages)}
        
        # Collect all samples with language labels
        self.samples = []
        
        for lang in languages:
            if lang not in datasets_by_language:
                logger.warning(f"Language {lang} not found in datasets")
                continue
                
            ds = datasets_by_language[lang]
            n_samples = len(ds)
            
            # Limit samples if specified
            if samples_per_language is not None:
                n_samples = min(n_samples, samples_per_language)
            
            # Add samples with language label
            indices = list(range(len(ds)))
            if samples_per_language is not None and len(indices) > samples_per_language:
                # Random subsample
                np.random.shuffle(indices)
                indices = indices[:samples_per_language]
            
            for idx in indices:
                self.samples.append({
                    "dataset": ds,
                    "index": idx,
                    "language": lang,
                    "label": self.lang_to_idx[lang],
                })
        
        # Balance classes if requested
        if balanced and samples_per_language is None:
            self._balance_classes()
        
        logger.info(f"Created LID dataset with {len(self.samples)} samples across {len(languages)} languages")
        self._log_class_distribution()
    
    def _balance_classes(self):
        """Balance classes by undersampling majority classes."""
        # Count samples per class
        class_counts = {}
        for sample in self.samples:
            label = sample["label"]
            class_counts[label] = class_counts.get(label, 0) + 1
        
        # Find minimum count
        min_count = min(class_counts.values())
        
        # Undersample each class to min_count
        balanced_samples = []
        samples_by_class = {i: [] for i in range(len(self.languages))}
        
        for sample in self.samples:
            samples_by_class[sample["label"]].append(sample)
        
        for label, samples in samples_by_class.items():
            np.random.shuffle(samples)
            balanced_samples.extend(samples[:min_count])
        
        self.samples = balanced_samples
        np.random.shuffle(self.samples)
    
    def _log_class_distribution(self):
        """Log class distribution."""
        class_counts = {}
        for sample in self.samples:
            lang = sample["language"]
            class_counts[lang] = class_counts.get(lang, 0) + 1
        
        logger.info(f"Class distribution: {class_counts}")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = self.samples[idx]
        
        # Get the actual data from the underlying dataset
        data = sample["dataset"][sample["index"]]
        
        return {
            "input_features": data["input_features"],
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "language": sample["language"],
        }


class LIDDataCollator:
    """Data collator for LID training."""
    
    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        # Stack input features
        input_features = torch.stack([f["input_features"] for f in features])
        
        # Stack labels
        labels = torch.stack([f["label"] for f in features])
        
        # Keep languages for debugging
        languages = [f["language"] for f in features]
        
        return {
            "input_features": input_features,
            "labels": labels,
            "languages": languages,
        }


# ============================================================================
# LID Trainer
# ============================================================================

class LIDTrainer:
    """Trainer for Language Identification classifier."""
    
    def __init__(
        self,
        classifier: LanguageClassifier,
        feature_extractor: EncoderFeatureExtractor,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        max_steps: int = 2000,
        eval_steps: int = 200,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "cosine",
        mixed_precision: str = "bf16",
        device: Optional[str] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        """
        Args:
            classifier: Language classifier model
            feature_extractor: Encoder feature extractor (frozen)
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            learning_rate: Peak learning rate
            weight_decay: Weight decay
            warmup_steps: Warmup steps
            max_steps: Maximum training steps
            eval_steps: Evaluate every N steps
            gradient_accumulation_steps: Gradient accumulation steps
            max_grad_norm: Max gradient norm for clipping
            scheduler_type: LR scheduler type
            mixed_precision: Mixed precision type
            device: Training device
            wandb_project: W&B project name
            wandb_run_name: W&B run name
        """
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        
        # Training config
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.scheduler_type = scheduler_type
        
        # Device and precision
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mixed_precision = mixed_precision
        
        # Setup autocast
        if mixed_precision == "fp16":
            self.scaler = torch.amp.GradScaler()
            self.autocast_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.scaler = None
            self.autocast_dtype = torch.bfloat16
        else:
            self.scaler = None
            self.autocast_dtype = torch.float32
        
        # Training state
        self.global_step = 0
        self.best_accuracy = 0.0
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup W&B
        self.wandb_run = None
        if wandb_project:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=wandb_run_name,
                    config={
                        "learning_rate": learning_rate,
                        "weight_decay": weight_decay,
                        "warmup_steps": warmup_steps,
                        "max_steps": max_steps,
                        "scheduler_type": scheduler_type,
                        "mixed_precision": mixed_precision,
                    }
                )
            except ImportError:
                logger.warning("wandb not installed, skipping logging")
    
    def _setup_optimizer(self):
        """Setup optimizer."""
        self.optimizer = AdamW(
            self.classifier.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        logger.info(f"Optimizer: AdamW with lr={self.learning_rate}")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        if self.scheduler_type == "linear":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            decay_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.0,
                total_iters=self.max_steps - self.warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.warmup_steps],
            )
        elif self.scheduler_type == "cosine":
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=self.max_steps - self.warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_type}")
        
        logger.info(f"Scheduler: {self.scheduler_type} with {self.warmup_steps} warmup")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def train(self) -> Dict[str, float]:
        """Run training loop."""
        logger.info("Starting LID classifier training...")
        
        # Move models to device
        self.classifier.to(self.device)
        self.classifier.train()
        
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()  # Always in eval mode (frozen)
        
        # Run initial evaluation
        # if self.eval_dataloader is not None:
        #     logger.info("Running initial evaluation...")
        #     initial_metrics = self.evaluate()
        #     logger.info(f"Initial accuracy: {initial_metrics['accuracy']:.4f}")
        
        train_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(total=self.max_steps, desc="Training LID")
        
        while self.global_step < self.max_steps:
            for batch in self.train_dataloader:
                # Training step
                loss = self._training_step(batch)
                train_loss += loss.item()
                num_batches += 1
                
                # Backward
                scaled_loss = loss / self.gradient_accumulation_steps
                
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Gradient accumulation
                if (num_batches % self.gradient_accumulation_steps) == 0:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.classifier.parameters(),
                            self.max_grad_norm,
                        )
                    
                    # Optimizer step
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "lr": f"{self.get_lr():.2e}",
                    })
                    
                    # Log to W&B
                    if self.wandb_run is not None and self.global_step % 10 == 0:
                        import wandb
                        wandb.log({
                            "train/loss": loss.item(),
                            "train/lr": self.get_lr(),
                            "train/step": self.global_step,
                        })
                    
                    # Evaluate
                    if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        metrics = self.evaluate()
                        self.classifier.train()
                        
                        # Log to W&B
                        if self.wandb_run is not None:
                            import wandb
                            wandb.log({
                                "eval/accuracy": metrics["accuracy"],
                                "eval/loss": metrics["loss"],
                                "eval/step": self.global_step,
                            })
                    
                    if self.global_step >= self.max_steps:
                        break
        
        pbar.close()
        
        # Final evaluation
        if self.eval_dataloader is not None:
            final_metrics = self.evaluate()
            logger.info(f"Final accuracy: {final_metrics['accuracy']:.4f}")
        else:
            final_metrics = {}
        
        final_metrics["train_loss"] = train_loss / max(num_batches, 1)
        final_metrics["global_step"] = self.global_step
        
        return final_metrics
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform single training step."""
        input_features = batch["input_features"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        with torch.autocast(
            device_type=self.device.split(":")[0],
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision != "no",
        ):
            # Extract encoder features (frozen)
            with torch.no_grad():
                encoder_features = self.feature_extractor(input_features)
            
            # Forward through classifier
            outputs = self.classifier(encoder_features, labels=labels)
            loss = outputs["loss"]
        
        return loss
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation."""
        logger.info(f"Evaluating at step {self.global_step}...")
        
        self.classifier.eval()
        
        all_predictions = []
        all_labels = []
        all_probs = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                with torch.autocast(
                    device_type=self.device.split(":")[0],
                    dtype=self.autocast_dtype,
                    enabled=self.mixed_precision != "no",
                ):
                    # Extract encoder features
                    encoder_features = self.feature_extractor(input_features)
                    
                    # Forward through classifier
                    outputs = self.classifier(encoder_features, labels=labels)
                
                total_loss += outputs["loss"].item()
                num_batches += 1
                
                # Get predictions
                probs = outputs["probs"]
                preds = probs.argmax(dim=-1)
                
                all_predictions.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
                all_probs.extend(probs.cpu().tolist())
        
        # Compute metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        accuracy = (all_predictions == all_labels).mean()
        
        # Per-class accuracy
        per_class_acc = {}
        for i, lang in enumerate(self.classifier.languages):
            mask = all_labels == i
            if mask.sum() > 0:
                per_class_acc[lang] = (all_predictions[mask] == all_labels[mask]).mean()
        
        # Confusion matrix
        num_classes = len(self.classifier.languages)
        confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
        for pred, label in zip(all_predictions, all_labels):
            confusion[label, pred] += 1
        
        metrics = {
            "accuracy": accuracy,
            "loss": total_loss / max(num_batches, 1),
            "per_class_accuracy": per_class_acc,
        }
        
        # Update best accuracy
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            metrics["is_best"] = True
        else:
            metrics["is_best"] = False
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"Per-class accuracy: {per_class_acc}")
        logger.info(f"Confusion matrix:\n{confusion}")
        
        return metrics
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save classifier checkpoint."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save classifier
        self.classifier.save(path / "classifier.pt")
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }
        torch.save(state, path / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {path}")


# ============================================================================
# Main
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train LID classifier for adapter routing")
    
    # Model arguments
    parser.add_argument(
        "--base_model",
        type=str,
        default="whisper-small",
        choices=["whisper-small", "whisper-medium", "whisper-large"],
        help="Base model for feature extraction",
    )
    parser.add_argument(
        "--encoder_layer",
        type=int,
        default=-1,
        help="Encoder layer to extract features from (-1 = last layer)",
    )
    
    # Classifier arguments
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[256, 128],
        help="Hidden layer dimensions for classifier",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "attention"],
        help="Pooling strategy",
    )
    parser.add_argument("--use_cnn", action="store_true", help="Use CNN for temporal modeling")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    
    # Data arguments
    parser.add_argument(
        "--languages",
        type=str,
        nargs="+",
        default=["hindi", "italian", "punjabi", "telugu"],
        help="Languages to train on",
    )
    parser.add_argument(
        "--data_sources",
        type=str,
        nargs="+",
        default=["common_voice"],
        help="Data sources to use",
    )
    parser.add_argument(
        "--samples_per_language",
        type=int,
        default=5000,
        help="Maximum samples per language",
    )
    parser.add_argument("--balanced", action="store_true", default=True, help="Balance classes by undersampling (not recommended with class_weights)")
    parser.add_argument("--no_balanced", action="store_true", help="Disable class balancing (use with --class_weights)")
    parser.add_argument(
        "--class_weights",
        type=str,
        default="none",
        choices=["none", "inverse_freq", "inverse_sqrt", "effective_samples"],
        help="Class weighting strategy for imbalanced data: "
             "none=no weighting, inverse_freq=1/frequency, "
             "inverse_sqrt=sqrt(max/count), effective_samples=CVPR2019 method",
    )
    parser.add_argument(
        "--class_weight_max",
        type=float,
        default=10.0,
        help="Maximum class weight to prevent extreme upweighting (default: 10.0)",
    )
    parser.add_argument(
        "--class_weight_smoothing",
        type=float,
        default=0.0,
        help="Smoothing factor to blend weights towards uniform. "
             "0=full weighting, 1=uniform. Recommended 0.3-0.5 for extreme imbalance.",
    )
    parser.add_argument("--max_duration", type=float, default=15.0, help="Max audio duration")
    parser.add_argument("--min_duration", type=float, default=1.0, help="Min audio duration")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    parser.add_argument("--max_steps", type=int, default=2000, help="Maximum training steps")
    parser.add_argument("--eval_steps", type=int, default=200, help="Evaluation interval")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation")
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="LR scheduler type",
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="bf16",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision",
    )
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    
    # W&B arguments
    parser.add_argument("--wandb_project", type=str, default="lid-classifier", help="W&B project")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="W&B run name")
    parser.add_argument("--no_wandb", action="store_true", help="Disable W&B")
    
    # Other
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
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
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load base model for feature extraction
    logger.info(f"Loading base model: {args.base_model}")
    base_model = load_base_model(args.base_model, device=device)
    processor = get_processor(args.base_model)
    
    # Create feature extractor
    feature_extractor = EncoderFeatureExtractor(
        model=base_model,
        layer_index=args.encoder_layer,
    )
    
    # Determine input dimension from encoder
    input_dim = feature_extractor.get_hidden_dim()
    logger.info(f"Encoder hidden dimension: {input_dim}")
    
    # Create classifier
    classifier = LanguageClassifier(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        num_classes=len(args.languages),
        dropout=args.dropout,
        pooling=args.pooling,
        use_cnn=args.use_cnn,
        label_smoothing=args.label_smoothing,
        languages=args.languages,
    )
    
    trainable_params = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    logger.info(f"Classifier trainable parameters: {trainable_params:,}")
    
    # Load datasets for each language
    logger.info("Loading datasets...")
    train_datasets = {}
    eval_datasets = {}
    
    for lang in args.languages:
        logger.info(f"Loading {lang} dataset...")
        try:
            train_datasets[lang] = create_dataset(
                language=lang,
                split="train",
                sources=args.data_sources,
                max_samples=args.samples_per_language,
                cache_dir=args.cache_dir,
                processor=processor,
                max_duration=args.max_duration,
                min_duration=args.min_duration,
            )
            
            eval_datasets[lang] = create_dataset(
                language=lang,
                split="validation",
                sources=args.data_sources,
                max_samples=args.samples_per_language // 5,  # 20% for eval
                cache_dir=args.cache_dir,
                processor=processor,
                max_duration=args.max_duration,
                min_duration=args.min_duration,
            )
            
            logger.info(f"  {lang}: {len(train_datasets[lang])} train, {len(eval_datasets[lang])} eval")
            
        except Exception as e:
            logger.error(f"Failed to load {lang}: {e}")
            continue
    
    if not train_datasets:
        raise ValueError("No datasets could be loaded!")
    
    # Count samples per language for class weighting
    train_class_counts = {lang: len(ds) for lang, ds in train_datasets.items()}
    logger.info(f"Training class counts: {train_class_counts}")
    
    # Determine if we should balance by undersampling
    use_balanced = args.balanced and not args.no_balanced and args.class_weights == "none"
    if args.class_weights != "none" and args.balanced and not args.no_balanced:
        logger.warning("Using --class_weights with --balanced is not recommended. "
                      "Consider using --no_balanced to keep all samples.")
    
    # Create LID datasets
    train_lid_dataset = LIDDataset(
        datasets_by_language=train_datasets,
        languages=args.languages,
        samples_per_language=args.samples_per_language,
        balanced=use_balanced,
    )
    
    # Compute and apply class weights if requested
    if args.class_weights != "none":
        # Get actual counts after dataset creation (may differ due to sampling)
        actual_counts = {}
        for sample in train_lid_dataset.samples:
            lang = sample["language"]
            actual_counts[lang] = actual_counts.get(lang, 0) + 1
        
        logger.info(f"Actual training class counts: {actual_counts}")
        
        class_weights = LanguageClassifier.compute_class_weights_from_counts(
            class_counts=actual_counts,
            languages=args.languages,
            strategy=args.class_weights,
            max_weight=args.class_weight_max,
            smoothing=args.class_weight_smoothing,
        )
        
        classifier.set_class_weights(class_weights.to(device))
        logger.info(f"Applied {args.class_weights} class weights (max={args.class_weight_max}, smoothing={args.class_weight_smoothing}): {dict(zip(args.languages, class_weights.tolist()))}")
    
    eval_lid_dataset = LIDDataset(
        datasets_by_language=eval_datasets,
        languages=args.languages,
        samples_per_language=args.samples_per_language // 5,
        balanced=True,
    )
    
    # Create data loaders
    collator = LIDDataCollator()
    
    train_dataloader = DataLoader(
        train_lid_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    eval_dataloader = DataLoader(
        eval_lid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    # Create trainer
    trainer = LIDTrainer(
        classifier=classifier,
        feature_extractor=feature_extractor,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        scheduler_type=args.scheduler_type,
        mixed_precision=args.mixed_precision,
        device=device,
        wandb_project=None if args.no_wandb else args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    # Train
    logger.info("Starting training...")
    final_metrics = trainer.train()
    
    # Save final model
    trainer.save_checkpoint(output_dir / "final")
    
    # Also save just the classifier for easy loading
    classifier.save(output_dir / "classifier.pt")
    
    logger.info(f"Training completed. Final metrics: {final_metrics}")
    logger.info(f"Best accuracy: {trainer.best_accuracy:.4f}")
    logger.info(f"Model saved to {output_dir}")
    
    return final_metrics


if __name__ == "__main__":
    main()
