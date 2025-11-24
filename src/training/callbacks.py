"""Training callbacks for logging, checkpointing, and early stopping."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class Callback(ABC):
    """Base callback class."""
    
    def on_train_begin(self, trainer: Any, **kwargs):
        """Called at the start of training."""
        pass
    
    def on_train_end(self, trainer: Any, **kwargs):
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, trainer: Any, epoch: int, **kwargs):
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, trainer: Any, epoch: int, metrics: Dict, **kwargs):
        """Called at the end of each epoch."""
        pass
    
    def on_step_begin(self, trainer: Any, step: int, **kwargs):
        """Called at the start of each training step."""
        pass
    
    def on_step_end(self, trainer: Any, step: int, loss: float, **kwargs):
        """Called at the end of each training step."""
        pass
    
    def on_evaluate_begin(self, trainer: Any, **kwargs):
        """Called at the start of evaluation."""
        pass
    
    def on_evaluate_end(self, trainer: Any, metrics: Dict, **kwargs):
        """Called at the end of evaluation."""
        pass


class WandbCallback(Callback):
    """Weights & Biases logging callback."""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_every: int = 50,
    ):
        """
        Args:
            project: W&B project name
            name: Run name
            config: Configuration to log
            log_every: Log every N steps
        """
        self.project = project
        self.name = name
        self.config = config or {}
        self.log_every = log_every
        self.run = None
        
    def on_train_begin(self, trainer: Any, **kwargs):
        """Initialize W&B run."""
        try:
            import wandb
            self.run = wandb.init(
                project=self.project,
                name=self.name,
                config=self.config,
                resume="allow",
            )
            logger.info(f"W&B run initialized: {self.run.name}")
        except ImportError:
            logger.warning("wandb not installed, skipping logging")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    def on_train_end(self, trainer: Any, **kwargs):
        """Finish W&B run."""
        if self.run is not None:
            import wandb
            wandb.finish()
    
    def on_step_end(self, trainer: Any, step: int, loss: float, **kwargs):
        """Log training step metrics."""
        if self.run is None or step % self.log_every != 0:
            return
            
        import wandb
        
        log_dict = {
            "train/loss": loss,
            "train/step": step,
            "train/learning_rate": trainer.get_lr(),
        }
        
        # Add any extra metrics
        log_dict.update(kwargs.get("extra_metrics", {}))
        
        wandb.log(log_dict, step=step)
    
    def on_evaluate_end(self, trainer: Any, metrics: Dict, **kwargs):
        """Log evaluation metrics."""
        if self.run is None:
            return
            
        import wandb
        
        log_dict = {f"eval/{k}": v for k, v in metrics.items()}
        wandb.log(log_dict, step=trainer.global_step)


class CheckpointCallback(Callback):
    """Checkpoint saving callback."""
    
    def __init__(
        self,
        save_dir: str,
        save_every: int = 1000,
        save_total_limit: int = 3,
        save_best_only: bool = True,
        metric_name: str = "wer",
        greater_is_better: bool = False,
    ):
        """
        Args:
            save_dir: Directory to save checkpoints
            save_every: Save every N steps
            save_total_limit: Maximum checkpoints to keep
            save_best_only: Only save when metric improves
            metric_name: Metric to track for best model
            greater_is_better: Whether higher metric is better
        """
        self.save_dir = Path(save_dir)
        self.save_every = save_every
        self.save_total_limit = save_total_limit
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.checkpoint_files = []
        
    def on_train_begin(self, trainer: Any, **kwargs):
        """Create checkpoint directory."""
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def on_step_end(self, trainer: Any, step: int, loss: float, **kwargs):
        """Save periodic checkpoint."""
        if step > 0 and step % self.save_every == 0:
            self._save_checkpoint(trainer, f"step_{step}")
    
    def on_evaluate_end(self, trainer: Any, metrics: Dict, **kwargs):
        """Save checkpoint if metric improved."""
        if not self.save_best_only:
            return
            
        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return
            
        is_better = (
            (self.greater_is_better and current_metric > self.best_metric) or
            (not self.greater_is_better and current_metric < self.best_metric)
        )
        
        if is_better:
            logger.info(
                f"New best {self.metric_name}: {current_metric:.4f} "
                f"(previous: {self.best_metric:.4f})"
            )
            self.best_metric = current_metric
            self._save_checkpoint(trainer, "best")
    
    def _save_checkpoint(self, trainer: Any, name: str):
        """Save checkpoint."""
        checkpoint_path = self.save_dir / name
        
        # Save model
        trainer.model.save_adapter(checkpoint_path)
        
        # Save training state
        state = {
            "global_step": trainer.global_step,
            "epoch": trainer.epoch,
            "best_metric": self.best_metric,
            "optimizer": trainer.optimizer.state_dict(),
        }
        if trainer.scheduler is not None:
            state["scheduler"] = trainer.scheduler.state_dict()
        
        torch.save(state, checkpoint_path / "training_state.pt")
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Track checkpoints for cleanup
        if name != "best":
            self.checkpoint_files.append(checkpoint_path)
            self._cleanup_old_checkpoints()
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if limit exceeded."""
        while len(self.checkpoint_files) > self.save_total_limit:
            old_checkpoint = self.checkpoint_files.pop(0)
            if old_checkpoint.exists():
                import shutil
                shutil.rmtree(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")


class EarlyStoppingCallback(Callback):
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.001,
        metric_name: str = "wer",
        greater_is_better: bool = False,
    ):
        """
        Args:
            patience: Number of evaluations without improvement before stopping
            min_delta: Minimum improvement to count as progress
            metric_name: Metric to track
            greater_is_better: Whether higher metric is better
        """
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        
        self.best_metric = float('-inf') if greater_is_better else float('inf')
        self.counter = 0
        self.should_stop = False
        
    def on_evaluate_end(self, trainer: Any, metrics: Dict, **kwargs):
        """Check if training should stop."""
        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return
        
        if self.greater_is_better:
            improved = current_metric > self.best_metric + self.min_delta
        else:
            improved = current_metric < self.best_metric - self.min_delta
        
        if improved:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
            logger.info(
                f"EarlyStopping: {self.counter}/{self.patience} "
                f"(best {self.metric_name}: {self.best_metric:.4f})"
            )
            
        if self.counter >= self.patience:
            self.should_stop = True
            logger.info(f"Early stopping triggered after {self.patience} evaluations")


class TensorBoardCallback(Callback):
    """TensorBoard logging callback."""
    
    def __init__(self, log_dir: str):
        """
        Args:
            log_dir: Directory for TensorBoard logs
        """
        self.log_dir = log_dir
        self.writer = None
        
    def on_train_begin(self, trainer: Any, **kwargs):
        """Initialize TensorBoard writer."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.log_dir)
            logger.info(f"TensorBoard logging to {self.log_dir}")
        except ImportError:
            logger.warning("tensorboard not installed, skipping logging")
    
    def on_train_end(self, trainer: Any, **kwargs):
        """Close TensorBoard writer."""
        if self.writer is not None:
            self.writer.close()
    
    def on_step_end(self, trainer: Any, step: int, loss: float, **kwargs):
        """Log training step."""
        if self.writer is not None:
            self.writer.add_scalar("train/loss", loss, step)
            self.writer.add_scalar("train/lr", trainer.get_lr(), step)
    
    def on_evaluate_end(self, trainer: Any, metrics: Dict, **kwargs):
        """Log evaluation metrics."""
        if self.writer is None:
            return
            
        for name, value in metrics.items():
            self.writer.add_scalar(f"eval/{name}", value, trainer.global_step)
