"""Data collation for speech-to-text models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch


@dataclass
class DataCollatorSpeechSeq2Seq:
    """Data collator for speech sequence-to-sequence models.
    
    Handles padding of input features and labels for batched training.
    """
    
    processor: Any
    decoder_start_token_id: int = None
    padding: Union[bool, str] = True
    max_length: int = None
    pad_to_multiple_of: int = None
    return_tensors: str = "pt"
    
    def __post_init__(self):
        if self.decoder_start_token_id is None:
            self.decoder_start_token_id = self.processor.tokenizer.bos_token_id
    
    def __call__(
        self, 
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Collate a list of features into a batch.
        
        Args:
            features: List of feature dictionaries with keys:
                - input_features: Log-mel spectrogram (time, n_mels)
                - labels: Tokenized target sequence
                
        Returns:
            Batched tensors with proper padding
        """
        # Separate input features and labels
        input_features = []
        labels = []
        
        for feature in features:
            input_features.append({"input_features": feature["input_features"]})
            labels.append(feature["labels"])
        
        # Pad input features (log-mel spectrograms)
        # Whisper expects fixed-length input (30s = 3000 frames)
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Pad labels with label_pad_token_id (-100 for ignore in loss)
        label_features = [{"input_ids": label} for label in labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        
        # Replace padding token id with -100 for loss computation
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        # Remove BOS token if present (model prepends it during generation)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        
        return batch


@dataclass  
class DataCollatorSpeechSeq2SeqWithLanguage(DataCollatorSpeechSeq2Seq):
    """Data collator that also handles language labels for routing."""
    
    language_to_id: Dict[str, int] = None
    
    def __call__(
        self,
        features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        """Collate features including language labels."""
        batch = super().__call__(features)
        
        # Add language labels if present
        if self.language_to_id is not None and "language" in features[0]:
            language_ids = [
                self.language_to_id.get(f["language"], 0) 
                for f in features
            ]
            batch["language_ids"] = torch.tensor(language_ids, dtype=torch.long)
        
        return batch


def create_collator(
    processor: Any,
    with_language: bool = False,
    language_to_id: Dict[str, int] = None,
    **kwargs
) -> Union[DataCollatorSpeechSeq2Seq, DataCollatorSpeechSeq2SeqWithLanguage]:
    """Factory function to create appropriate data collator.
    
    Args:
        processor: Whisper processor
        with_language: Whether to include language labels
        language_to_id: Mapping from language names to IDs
        **kwargs: Additional arguments for collator
        
    Returns:
        Configured data collator
    """
    if with_language:
        return DataCollatorSpeechSeq2SeqWithLanguage(
            processor=processor,
            language_to_id=language_to_id,
            **kwargs
        )
    return DataCollatorSpeechSeq2Seq(processor=processor, **kwargs)
