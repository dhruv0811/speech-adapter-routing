"""Base model utilities."""

import logging
from typing import Optional, Tuple, Union

import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperConfig,
)

logger = logging.getLogger(__name__)

# Model name mappings
MODEL_NAME_MAP = {
    "whisper-small": "openai/whisper-small",
    "whisper-medium": "openai/whisper-medium",
    "whisper-large": "openai/whisper-large-v3",
    "whisper-large-v2": "openai/whisper-large-v2",
    "whisper-large-v3": "openai/whisper-large-v3",
    "whisper-tiny": "openai/whisper-tiny",
    "whisper-base": "openai/whisper-base",
}

# Language code mappings for Whisper
LANGUAGE_CODES = {
    "hindi": "hi",
    "italian": "it", 
    "punjabi": "pa",
    "telugu": "te",
    "english": "en",
    "german": "de",
    "french": "fr",
    "spanish": "es",
}


def get_model_name(model_id: str) -> str:
    """Get full model name from short ID."""
    return MODEL_NAME_MAP.get(model_id, model_id)


def get_processor(
    model_name: str,
    language: Optional[str] = None,
    task: str = "transcribe",
    cache_dir: Optional[str] = None,
) -> WhisperProcessor:
    """Load Whisper processor.
    
    Args:
        model_name: Model name or ID
        language: Optional language for tokenizer
        task: Task type (transcribe or translate)
        cache_dir: Directory to cache model
        
    Returns:
        WhisperProcessor instance
    """
    model_name = get_model_name(model_name)
    
    # Map language name to code
    if language:
        language = LANGUAGE_CODES.get(language.lower(), language)
    
    processor = WhisperProcessor.from_pretrained(
        model_name,
        language=language,
        task=task,
        cache_dir=cache_dir,
    )
    
    return processor


def load_base_model(
    model_name: str,
    device: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    cache_dir: Optional[str] = None,
    use_flash_attention: bool = False,
) -> WhisperForConditionalGeneration:
    """Load base Whisper model.
    
    Args:
        model_name: Model name or ID
        device: Device to load model on
        dtype: Data type for model weights
        cache_dir: Directory to cache model
        use_flash_attention: Whether to use Flash Attention 2
        
    Returns:
        WhisperForConditionalGeneration model
    """
    model_name = get_model_name(model_name)
    
    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Determine dtype
    if dtype is None:
        if device == "cuda":
            # Use bf16 on Ampere+ GPUs, fp16 otherwise
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16
        else:
            dtype = torch.float32
    
    logger.info(f"Loading {model_name} to {device} with dtype {dtype}")
    
    # Load model
    model_kwargs = {
        "cache_dir": cache_dir,
        "torch_dtype": dtype,
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = WhisperForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )
    
    # Clear forced decoder IDs for multilingual
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    
    model.to(device)
    
    # Log model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded model with {total_params / 1e6:.1f}M parameters")
    
    return model


def get_model_info(model_name: str) -> dict:
    """Get model configuration info.
    
    Args:
        model_name: Model name or ID
        
    Returns:
        Dictionary with model configuration
    """
    model_name = get_model_name(model_name)
    config = WhisperConfig.from_pretrained(model_name)
    
    return {
        "name": model_name,
        "hidden_size": config.d_model,
        "encoder_layers": config.encoder_layers,
        "decoder_layers": config.decoder_layers,
        "encoder_attention_heads": config.encoder_attention_heads,
        "decoder_attention_heads": config.decoder_attention_heads,
        "encoder_ffn_dim": config.encoder_ffn_dim,
        "decoder_ffn_dim": config.decoder_ffn_dim,
        "vocab_size": config.vocab_size,
        "max_source_positions": config.max_source_positions,
        "max_target_positions": config.max_target_positions,
    }
