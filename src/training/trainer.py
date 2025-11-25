"""Main trainer class for ASR LoRA fine-tuning."""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
    ConstantLR,
)
from tqdm import tqdm

from .callbacks import Callback, EarlyStoppingCallback
from .metrics import compute_metrics

logger = logging.getLogger(__name__)


class ASRTrainer:
    """Trainer for ASR models with LoRA adapters."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
        processor: Any = None,
        learning_rate: float = 5e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        max_steps: int = 5000,
        eval_steps: int = 500,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        scheduler_type: str = "linear",
        mixed_precision: str = "bf16",
        callbacks: Optional[List[Callback]] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model: Model to train
            train_dataloader: Training data loader
            eval_dataloader: Evaluation data loader
            processor: Whisper processor for decoding
            learning_rate: Peak learning rate
            weight_decay: Weight decay for optimizer
            warmup_steps: Number of warmup steps
            max_steps: Maximum training steps
            eval_steps: Evaluate every N steps
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_grad_norm: Maximum gradient norm for clipping
            scheduler_type: LR scheduler type (linear, cosine, constant)
            mixed_precision: Mixed precision type (fp16, bf16, no)
            callbacks: List of callbacks
            device: Device to train on
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.processor = processor
        
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
        
        # Setup mixed precision
        self.scaler = None
        if mixed_precision == "fp16":
            self.scaler = torch.amp.GradScaler()
            self.autocast_dtype = torch.float16
        elif mixed_precision == "bf16":
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = torch.float32
        
        # Callbacks
        self.callbacks = callbacks or []
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.optimizer = None
        self.scheduler = None
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
    def _setup_optimizer(self):
        """Setup optimizer with weight decay."""
        # Separate weight decay for different parameter groups
        decay_params = []
        no_decay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "LayerNorm" in name or "layer_norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        
        optimizer_groups = [
            {"params": decay_params, "weight_decay": self.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        
        self.optimizer = AdamW(
            optimizer_groups,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        logger.info(f"Optimizer: AdamW with lr={self.learning_rate}")
        
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        total_steps = self.max_steps
        
        if self.scheduler_type == "linear":
            # Linear warmup then linear decay
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
                total_iters=total_steps - self.warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, decay_scheduler],
                milestones=[self.warmup_steps],
            )
            
        elif self.scheduler_type == "cosine":
            # Linear warmup then cosine decay
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - self.warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[self.warmup_steps],
            )
            
        elif self.scheduler_type == "constant":
            # Linear warmup then constant
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=1e-8,
                end_factor=1.0,
                total_iters=self.warmup_steps,
            )
            constant_scheduler = ConstantLR(
                self.optimizer,
                factor=1.0,
                total_iters=total_steps - self.warmup_steps,
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, constant_scheduler],
                milestones=[self.warmup_steps],
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.scheduler_type}")
            
        logger.info(f"Scheduler: {self.scheduler_type} with {self.warmup_steps} warmup steps")
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]["lr"]
    
    def train(self) -> Dict[str, float]:
        """Run training loop.
        
        Returns:
            Final training metrics
        """
        logger.info("Starting training...")
        
        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        self.model.train()
        self.model.to(self.device)
        
        train_loss = 0.0
        num_batches = 0
        
        # Progress bar
        pbar = tqdm(total=self.max_steps, desc="Training")
        pbar.update(self.global_step)
        
        # Training loop
        while self.global_step < self.max_steps:
            self.epoch += 1
            
            for callback in self.callbacks:
                callback.on_epoch_begin(self, self.epoch)
            
            for batch_idx, batch in enumerate(self.train_dataloader):
                # Trigger step callbacks
                for callback in self.callbacks:
                    callback.on_step_begin(self, self.global_step)
                
                # Forward pass
                loss = self._training_step(batch)
                loss_value = loss.item()
                train_loss += loss_value
                num_batches += 1
                
                # Backward pass
                scaled_loss = loss / self.gradient_accumulation_steps
                
                if self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        if self.scaler is not None:
                            self.scaler.unscale_(self.optimizer)
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(),
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
                        "loss": f"{loss_value:.4f}",
                        "lr": f"{self.get_lr():.2e}",
                    })
                    
                    # Trigger step callbacks
                    for callback in self.callbacks:
                        callback.on_step_end(self, self.global_step, loss_value)
                    
                    # Evaluate
                    if self.eval_dataloader is not None and self.global_step % self.eval_steps == 0:
                        metrics = self.evaluate()
                        self.model.train()
                        
                        # Check early stopping
                        for callback in self.callbacks:
                            if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                                logger.info("Early stopping triggered")
                                pbar.close()
                                return self._finalize_training(train_loss, num_batches)
                    
                    # Check if done
                    if self.global_step >= self.max_steps:
                        break
            
            # Epoch end callbacks
            epoch_metrics = {"train_loss": train_loss / max(num_batches, 1)}
            for callback in self.callbacks:
                callback.on_epoch_end(self, self.epoch, epoch_metrics)
        
        pbar.close()
        return self._finalize_training(train_loss, num_batches)
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Loss tensor (for backward pass)
        """
        # Move batch to device
        input_features = batch["input_features"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        # Forward pass with mixed precision
        with torch.autocast(
            device_type=self.device.split(":")[0],
            dtype=self.autocast_dtype,
            enabled=self.mixed_precision != "no",
        ):
            outputs = self.model(
                input_features=input_features,
                labels=labels,
            )
            loss = outputs.loss
        
        return loss
    
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation.
        
        Returns:
            Evaluation metrics
        """
        logger.info(f"Evaluating at step {self.global_step}...")
        
        for callback in self.callbacks:
            callback.on_evaluate_begin(self)
        
        self.model.eval()
        
        all_predictions = []
        all_references = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Compute loss
                with torch.autocast(
                    device_type=self.device.split(":")[0],
                    dtype=self.autocast_dtype,
                    enabled=self.mixed_precision != "no",
                ):
                    outputs = self.model(
                        input_features=input_features,
                        labels=labels,
                    )
                    total_loss += outputs.loss.item()
                    num_batches += 1
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_features=input_features,
                    max_new_tokens=256,
                )
                
                # Decode predictions
                predictions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                
                # Decode references
                # Replace -100 with pad token for decoding
                labels = labels.clone()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(
                    labels,
                    skip_special_tokens=True,
                )
                
                all_predictions.extend(predictions)
                all_references.extend(references)
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_references)
        metrics["loss"] = total_loss / max(num_batches, 1)
        
        logger.info(f"Evaluation results: WER={metrics['wer']:.4f}, CER={metrics['cer']:.4f}")
        
        # Trigger callbacks
        for callback in self.callbacks:
            callback.on_evaluate_end(self, metrics)
        
        return metrics
    
    def _finalize_training(
        self,
        train_loss: float,
        num_batches: int,
    ) -> Dict[str, float]:
        """Finalize training and return metrics.
        
        Args:
            train_loss: Total training loss
            num_batches: Number of batches processed
            
        Returns:
            Final metrics
        """
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        final_metrics = {
            "train_loss": train_loss / max(num_batches, 1),
            "global_step": self.global_step,
            "epochs": self.epoch,
        }
        
        logger.info(f"Training completed. Final metrics: {final_metrics}")
        
        return final_metrics
    
    def save_checkpoint(self, path: Union[str, Path]):
        """Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_adapter(path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
        }
        torch.save(state, path / "training_state.pt")
        
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: Union[str, Path]):
        """Load training checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        path = Path(path)
        
        # Load training state
        state = torch.load(path / "training_state.pt", map_location=self.device)
        
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.optimizer.load_state_dict(state["optimizer"])
        
        if self.scheduler and state.get("scheduler"):
            self.scheduler.load_state_dict(state["scheduler"])
        
        logger.info(f"Checkpoint loaded from {path} (step {self.global_step})")
