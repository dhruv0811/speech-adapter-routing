"""Dataset loaders for multilingual ASR training."""

import logging
from typing import Dict, List, Optional, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset
from datasets import Audio, concatenate_datasets, Dataset as HFDataset, DatasetDict
from datasets import load_dataset as hf_load_dataset
from transformers import WhisperProcessor

logger = logging.getLogger(__name__)


class ASRDataset(Dataset):
    """PyTorch Dataset wrapper for ASR data."""
    
    def __init__(
        self,
        hf_dataset,
        processor: WhisperProcessor,
        audio_column: str = "audio",
        text_column: str = "text",
        language: str = "en",
        max_duration: float = 30.0,
        min_duration: float = 1.0,
        max_label_length: int = 448,  # Whisper's max token length
        sampling_rate: int = 16000,
    ):
        """
        Args:
            hf_dataset: HuggingFace dataset object
            processor: Whisper processor for tokenization
            audio_column: Name of audio column in dataset
            text_column: Name of text column in dataset  
            language: Language code for tokenizer
            max_duration: Maximum audio duration in seconds
            min_duration: Minimum audio duration in seconds
            max_label_length: Maximum label length in tokens (Whisper max is 448)
            sampling_rate: Target sampling rate
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.audio_column = audio_column
        self.text_column = text_column
        self.language = language
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.max_label_length = max_label_length
        self.sampling_rate = sampling_rate
        
        # Filter by duration if audio info available
        self._filter_by_duration()
        
        # Filter by label length to avoid exceeding Whisper's max
        self._filter_by_label_length()
        
    def _filter_by_duration(self):
        """Filter samples by audio duration."""
        def is_valid_duration(example):
            try:
                audio = example[self.audio_column]
                if isinstance(audio, dict) and "array" in audio:
                    duration = len(audio["array"]) / audio.get("sampling_rate", self.sampling_rate)
                    return self.min_duration <= duration <= self.max_duration
                return True  # Keep if we can't determine duration
            except Exception:
                return True
                
        original_size = len(self.dataset)
        self.dataset = self.dataset.filter(is_valid_duration)
        filtered_size = len(self.dataset)
        
        if filtered_size < original_size:
            logger.info(f"Filtered {original_size - filtered_size} samples by duration. "
                       f"Remaining: {filtered_size}")
    
    def _filter_by_label_length(self):
        """Filter samples where transcription exceeds max token length."""
        def is_valid_label_length(example):
            try:
                text = example[self.text_column]
                if text is None:
                    return True
                text = str(text).strip()
                # Tokenize WITH special tokens to match actual training tokenization
                tokens = self.processor.tokenizer(text, add_special_tokens=True).input_ids
                return len(tokens) <= self.max_label_length
            except Exception:
                return True
        
        original_size = len(self.dataset)
        self.dataset = self.dataset.filter(is_valid_label_length)
        filtered_size = len(self.dataset)
        
        if filtered_size < original_size:
            logger.info(f"Filtered {original_size - filtered_size} samples by label length (>{self.max_label_length} tokens). "
                       f"Remaining: {filtered_size}")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        example = self.dataset[idx]
        
        # Extract audio
        audio = example[self.audio_column]
        if isinstance(audio, dict):
            audio_array = audio["array"]
            sr = audio.get("sampling_rate", self.sampling_rate)
        else:
            audio_array = audio
            sr = self.sampling_rate
            
        # Extract text
        text = example[self.text_column]
        if text is None:
            text = ""
        text = str(text).strip()
        
        # Process audio to log-mel spectrogram
        input_features = self.processor.feature_extractor(
            audio_array,
            sampling_rate=sr,
            return_tensors="pt",
        ).input_features[0]
        
        # Tokenize text
        labels = self.processor.tokenizer(
            text,
            return_tensors="pt",
        ).input_ids[0]
        
        return {
            "input_features": input_features,
            "labels": labels,
            "text": text,  # Keep original text for debugging
        }


def load_common_voice(
    language: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> HFDataset:
    """Load Common Voice 17.0 dataset from HuggingFace.
    
    Uses fsicoli/common_voice_17_0 which has all languages including Punjabi.
    Requires trust_remote_code=True for the loading script.
    
    Args:
        language: Language code (e.g., "hi", "it", "pa-IN", "te")
        split: Dataset split (train, validation, test)
        cache_dir: Directory to cache dataset
        streaming: Whether to stream dataset (recommended for large datasets)
        
    Returns:
        HuggingFace dataset
    """
    # Normalize language codes for Common Voice
    # CV uses specific codes, map common variations
    lang_code_map = {
        "hi": "hi",
        "hindi": "hi",
        "it": "it",
        "italian": "it",
        "pa-IN": "pa-IN",
        "pa": "pa-IN",
        "punjabi": "pa-IN",
        "te": "te",
        "telugu": "te",
        "en": "en",
        "english": "en",
        "de": "de",
        "german": "de",
        "fr": "fr",
        "french": "fr",
        "es": "es",
        "spanish": "es",
    }
    
    lang_code = lang_code_map.get(language.lower(), language)
    
    logger.info(f"Loading Common Voice 17.0 for '{lang_code}' from HuggingFace (fsicoli/common_voice_17_0), split={split}")
    
    try:
        dataset = hf_load_dataset(
            "fsicoli/common_voice_17_0",
            lang_code,
            split=split,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,  # Required for fsicoli dataset loading script
        )
        
        # Cast audio to 16kHz for Whisper compatibility
        if not streaming:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Rename 'sentence' to 'text' for consistency across datasets
        if "sentence" in dataset.column_names:
            dataset = dataset.rename_column("sentence", "text")
        
        if streaming:
            logger.info(f"Loaded Common Voice 17.0 ({lang_code}) in streaming mode")
        else:
            logger.info(f"Loaded {len(dataset)} samples from Common Voice 17.0 ({lang_code})")
        
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load Common Voice 17.0 for '{lang_code}': {e}")
        raise


def load_ai4bharat(
    language: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> DatasetDict:
    """Load AI4Bharat IndicVoices dataset.
    
    Dataset: ai4bharat/IndicVoices
    Available languages: assamese, bengali, bodo, dogri, gujarati, hindi, kannada,
                        kashmiri, konkani, maithili, malayalam, manipuri, marathi,
                        nepali, odia, punjabi, sanskrit, santali, sindhi, tamil,
                        telugu, urdu
    
    Args:
        language: Language name (e.g., "hindi", "punjabi", "telugu")
        split: Dataset split
        cache_dir: Directory to cache dataset
        streaming: Whether to stream dataset
        
    Returns:
        HuggingFace dataset
    """
    logger.info(f"Loading AI4Bharat IndicVoices for {language}, split={split}")
    
    # Map common variations to IndicVoices language names
    lang_map = {
        "hi": "hindi",
        "hindi": "hindi",
        "pa": "punjabi",
        "punjabi": "punjabi", 
        "te": "telugu",
        "telugu": "telugu",
        "ta": "tamil",
        "tamil": "tamil",
        "bn": "bengali",
        "bengali": "bengali",
        "mr": "marathi",
        "marathi": "marathi",
        "gu": "gujarati",
        "gujarati": "gujarati",
        "kn": "kannada",
        "kannada": "kannada",
        "ml": "malayalam",
        "malayalam": "malayalam",
        "or": "odia",
        "odia": "odia",
        "as": "assamese",
        "assamese": "assamese",
        "ur": "urdu",
        "urdu": "urdu",
        "ne": "nepali",
        "nepali": "nepali",
        "sa": "sanskrit",
        "sanskrit": "sanskrit",
        "ks": "kashmiri",
        "kashmiri": "kashmiri",
        "sd": "sindhi",
        "sindhi": "sindhi",
        "doi": "dogri",
        "dogri": "dogri",
        "kok": "konkani",
        "konkani": "konkani",
        "mai": "maithili",
        "maithili": "maithili",
        "mni": "manipuri",
        "manipuri": "manipuri",
        "sat": "santali",
        "santali": "santali",
        "brx": "bodo",
        "bodo": "bodo",
    }
    
    lang_name = lang_map.get(language.lower(), language.lower())
    
    # Map split names: IndicVoices uses 'valid' not 'validation'
    split_map = {
        "validation": "valid",
        "val": "valid",
        "dev": "valid",
    }
    actual_split = split_map.get(split, split)
    
    try:
        dataset = hf_load_dataset(
            "ai4bharat/IndicVoices",
            lang_name,
            split=actual_split,
            cache_dir=cache_dir,
            streaming=streaming,
            trust_remote_code=True,
        )
        
        # IndicVoices uses 'audio_filepath' column, rename to 'audio' for consistency
        if "audio_filepath" in dataset.column_names:
            dataset = dataset.rename_column("audio_filepath", "audio")
        
        # Cast audio to correct sampling rate  
        if not streaming:
            dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        
        # Rename text columns to standard 'text' if needed
        # IndicVoices has 'verbatim' and 'normalized' - use 'normalized' as the text
        if "text" not in dataset.column_names:
            if "normalized" in dataset.column_names:
                dataset = dataset.rename_column("normalized", "text")
            elif "verbatim" in dataset.column_names:
                dataset = dataset.rename_column("verbatim", "text")
            elif "transcription" in dataset.column_names:
                dataset = dataset.rename_column("transcription", "text")
            elif "sentence" in dataset.column_names:
                dataset = dataset.rename_column("sentence", "text")
            
        if streaming:
            logger.info(f"Loaded AI4Bharat IndicVoices ({lang_name}) in streaming mode")
        else:
            logger.info(f"Loaded {len(dataset)} samples from AI4Bharat IndicVoices ({lang_name})")
            
        return dataset
        
    except Exception as e:
        logger.error(f"Failed to load AI4Bharat IndicVoices for '{lang_name}': {e}")
        raise


def load_mls(
    language: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> DatasetDict:
    """Load Multilingual LibriSpeech (MLS) dataset.
    
    Args:
        language: Language name (e.g., "italian", "german", "french")
        split: Dataset split
        cache_dir: Directory to cache dataset
        streaming: Whether to stream dataset
        
    Returns:
        HuggingFace dataset
    """
    logger.info(f"Loading MLS for {language}, split={split}")
    
    dataset = hf_load_dataset(
        "facebook/multilingual_librispeech",
        language.lower(),
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    
    # Cast audio to correct sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    logger.info(f"Loaded {len(dataset) if not streaming else 'streaming'} samples")
    return dataset


def load_fleurs(
    language: str,
    split: str = "train",
    cache_dir: Optional[str] = None,
    streaming: bool = False,
) -> DatasetDict:
    """Load FLEURS dataset.
    
    Args:
        language: FLEURS language code (e.g., "hi_in", "it_it", "pa_in", "te_in")
        split: Dataset split
        cache_dir: Directory to cache dataset
        streaming: Whether to stream dataset
        
    Returns:
        HuggingFace dataset
    """
    logger.info(f"Loading FLEURS for {language}, split={split}")
    
    dataset = hf_load_dataset(
        "google/fleurs",
        language,
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
    )
    
    # Cast audio to correct sampling rate
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    
    # Rename columns to standard names
    if "transcription" in dataset.column_names:
        dataset = dataset.rename_column("transcription", "text")
        
    logger.info(f"Loaded {len(dataset) if not streaming else 'streaming'} samples")
    return dataset


def create_dataset(
    language: str,
    split: str = "train",
    sources: List[str] = ["common_voice"],
    mixing_ratios: Optional[Dict[str, float]] = None,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    processor: Optional[WhisperProcessor] = None,
    **kwargs,
) -> Union[Dataset, ASRDataset]:
    """Create a combined dataset from multiple sources.
    
    Args:
        language: Target language
        split: Dataset split
        sources: List of dataset sources to use
        mixing_ratios: Optional dict of source->ratio for mixing
        max_samples: Maximum total samples
        cache_dir: Directory to cache datasets
        processor: Optional Whisper processor (if provided, returns ASRDataset)
        **kwargs: Additional arguments passed to ASRDataset
        
    Returns:
        Combined dataset (ASRDataset if processor provided, else HF dataset)
    """
    # Language code mappings
    lang_configs = {
        "hindi": {
            "common_voice": "hi",
            "ai4bharat": "hindi",
            "fleurs": "hi_in",
        },
        "italian": {
            "common_voice": "it",
            "mls": "italian",
            "fleurs": "it_it",
        },
        "punjabi": {
            "common_voice": "pa-IN",
            "ai4bharat": "punjabi",
            "fleurs": "pa_in",
        },
        "telugu": {
            "common_voice": "te",
            "ai4bharat": "telugu",
            "fleurs": "te_in",
        },
    }
    
    lang_config = lang_configs.get(language.lower(), {})
    
    # Load datasets from each source
    datasets_to_combine = []
    source_sizes = {}
    
    for source in sources:
        try:
            if source == "common_voice" and "common_voice" in lang_config:
                ds = load_common_voice(
                    lang_config["common_voice"], 
                    split=split, 
                    cache_dir=cache_dir
                )
            elif source == "ai4bharat" and "ai4bharat" in lang_config:
                ds = load_ai4bharat(
                    lang_config["ai4bharat"],
                    split=split,
                    cache_dir=cache_dir
                )
            elif source == "mls" and "mls" in lang_config:
                ds = load_mls(
                    lang_config["mls"],
                    split=split,
                    cache_dir=cache_dir
                )
            elif source == "fleurs" and "fleurs" in lang_config:
                ds = load_fleurs(
                    lang_config["fleurs"],
                    split=split,
                    cache_dir=cache_dir
                )
            else:
                logger.warning(f"Source {source} not available for {language}")
                continue
                
            source_sizes[source] = len(ds)
            datasets_to_combine.append((source, ds))
            
        except Exception as e:
            logger.warning(f"Failed to load {source} for {language}: {e}")
            continue
    
    if not datasets_to_combine:
        raise ValueError(f"No datasets could be loaded for {language}")
    
    # Apply mixing ratios if provided
    if mixing_ratios and len(datasets_to_combine) > 1:
        mixed_datasets = []
        for source, ds in datasets_to_combine:
            ratio = mixing_ratios.get(source, 1.0 / len(datasets_to_combine))
            n_samples = int(len(ds) * ratio)
            if n_samples > 0:
                ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))
                mixed_datasets.append(ds)
        combined = concatenate_datasets(mixed_datasets)
    else:
        combined = concatenate_datasets([ds for _, ds in datasets_to_combine])
    
    # Limit total samples if specified
    if max_samples and len(combined) > max_samples:
        combined = combined.shuffle(seed=42).select(range(max_samples))
    
    logger.info(f"Created combined dataset with {len(combined)} samples from {list(source_sizes.keys())}")
    
    # Wrap in ASRDataset if processor provided
    if processor is not None:
        return ASRDataset(
            combined,
            processor=processor,
            audio_column="audio",
            text_column="text",
            language=language,
            **kwargs,
        )
    
    return combined
