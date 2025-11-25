"""Whisper model with LoRA adapters."""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
import torch.nn as nn
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import (
    LoraConfig,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training,
)

from .base import load_base_model, get_processor, get_model_name

logger = logging.getLogger(__name__)


class WhisperLoRA(nn.Module):
    """Whisper model with LoRA adapters for efficient fine-tuning."""
    
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Optional[List[str]] = None,
        language: Optional[str] = None,
        task: str = "transcribe",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        cache_dir: Optional[str] = None,
        use_gradient_checkpointing: bool = True,
    ):
        """
        Args:
            model_name: Base Whisper model name
            lora_r: LoRA rank
            lora_alpha: LoRA alpha (scale factor)
            lora_dropout: Dropout probability for LoRA layers
            target_modules: List of modules to apply LoRA to
            language: Target language for processor
            task: Task type (transcribe/translate)
            device: Device to load model on
            dtype: Data type for model
            cache_dir: Directory to cache model
            use_gradient_checkpointing: Whether to use gradient checkpointing
        """
        super().__init__()
        
        self.model_name = get_model_name(model_name)
        self.language = language
        self.task = task
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default target modules for Whisper
        if target_modules is None:
            target_modules = ["q_proj", "v_proj"]
        
        # Load processor
        self.processor = get_processor(
            self.model_name,
            language=language,
            task=task,
            cache_dir=cache_dir,
        )
        
        # Load base model
        self.model = load_base_model(
            self.model_name,
            device=self.device,
            dtype=dtype,
            cache_dir=cache_dir,
        )
        
        # Enable gradient checkpointing
        if use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
        
        # Configure LoRA
        # Note: Don't use TaskType.SEQ_2_SEQ_LM as it causes PEFT to inject input_ids handling
        # which conflicts with Whisper's input_features interface
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            # Removed task_type to avoid PEFT's seq2seq wrapper adding input_ids
        )
        
        # Apply LoRA
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Log trainable parameters
        self._log_trainable_params()
        
    def _log_trainable_params(self):
        """Log trainable parameter count."""
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_pct = 100 * trainable_params / total_params
        
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({trainable_pct:.2f}%)"
        )
        
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            input_features: Log-mel spectrogram (batch, n_mels, time)
            labels: Target token IDs (batch, seq_len)
            attention_mask: Attention mask for encoder
            decoder_input_ids: Decoder input IDs (optional, auto-generated from labels)
            decoder_attention_mask: Decoder attention mask
            **kwargs: Additional arguments (ignored to avoid conflicts)
            
        Returns:
            Model outputs including loss if labels provided
        """
        # Only pass supported arguments to avoid PEFT/Whisper conflicts
        return self.model(
            input_features=input_features,
            labels=labels,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
        )
    
    def generate(
        self,
        input_features: torch.Tensor,
        max_new_tokens: int = 256,
        num_beams: int = 1,
        language: Optional[str] = None,
        task: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate transcription.
        
        Args:
            input_features: Log-mel spectrogram
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width
            language: Override language
            task: Override task
            **kwargs: Additional generation arguments
            
        Returns:
            Generated token IDs
        """
        # Disable gradient checkpointing for generation
        was_checkpointing = self.model.base_model.model.model.encoder.gradient_checkpointing
        if was_checkpointing:
            self.model.base_model.model.gradient_checkpointing_disable()
            self.model.config.use_cache = True
        
        try:
            generated_ids = self.model.generate(
                input_features=input_features,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                **kwargs,
            )
        finally:
            # Restore gradient checkpointing
            if was_checkpointing:
                self.model.base_model.model.gradient_checkpointing_enable()
                self.model.config.use_cache = False
        
        return generated_ids
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
    ) -> List[str]:
        """Decode token IDs to text.
        
        Args:
            token_ids: Token IDs to decode
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            List of decoded strings
        """
        return self.processor.batch_decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
        )
    
    def save_adapter(self, save_path: Union[str, Path]):
        """Save LoRA adapter weights.
        
        Args:
            save_path: Directory to save adapter
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        logger.info(f"Saved adapter to {save_path}")
        
    def load_adapter(self, adapter_path: Union[str, Path]):
        """Load LoRA adapter weights.
        
        Args:
            adapter_path: Path to adapter directory
        """
        adapter_path = Path(adapter_path)
        
        # Load adapter weights
        self.model = PeftModel.from_pretrained(
            self.model.base_model,
            adapter_path,
        )
        logger.info(f"Loaded adapter from {adapter_path}")
        
    def merge_and_unload(self) -> WhisperForConditionalGeneration:
        """Merge LoRA weights into base model and unload adapter.
        
        Returns:
            Merged model without adapters
        """
        return self.model.merge_and_unload()
    
    def train(self, mode: bool = True):
        """Set training mode."""
        self.model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode."""
        self.model.eval()
        return self


def create_whisper_lora(
    model_name: str,
    lora_config: Optional[Dict] = None,
    language: Optional[str] = None,
    **kwargs,
) -> WhisperLoRA:
    """Factory function to create WhisperLoRA model.
    
    Args:
        model_name: Model name or ID
        lora_config: LoRA configuration dictionary
        language: Target language
        **kwargs: Additional arguments
        
    Returns:
        Configured WhisperLoRA model
    """
    lora_config = lora_config or {}
    
    return WhisperLoRA(
        model_name=model_name,
        lora_r=lora_config.get("r", 16),
        lora_alpha=lora_config.get("lora_alpha", 32),
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        target_modules=lora_config.get("target_modules", ["q_proj", "v_proj"]),
        language=language,
        **kwargs,
    )


def load_whisper_lora_from_checkpoint(
    checkpoint_path: Union[str, Path],
    model_name: str,
    language: Optional[str] = None,
    device: Optional[str] = None,
    **kwargs,
) -> WhisperLoRA:
    """Load WhisperLoRA model from checkpoint.
    
    Args:
        checkpoint_path: Path to adapter checkpoint
        model_name: Base model name
        language: Target language
        device: Device to load on
        **kwargs: Additional arguments
        
    Returns:
        Loaded WhisperLoRA model
    """
    checkpoint_path = Path(checkpoint_path)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load base model
    base_model = load_base_model(model_name, device=device, **kwargs)
    
    # Load adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.to(device)
    
    # Get processor
    processor = get_processor(model_name, language=language)
    
    # Wrap in WhisperLoRA-like interface
    wrapper = WhisperLoRA.__new__(WhisperLoRA)
    wrapper.model = model
    wrapper.processor = processor
    wrapper.model_name = model_name
    wrapper.language = language
    wrapper.device = device
    
    logger.info(f"Loaded WhisperLoRA from {checkpoint_path}")
    
    return wrapper
