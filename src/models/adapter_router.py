"""Adapter routing for language detection and selection."""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LanguageClassifier(nn.Module):
    """Lightweight language classifier for adapter routing.
    
    Takes encoder features and predicts language probabilities.
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
    ):
        """
        Args:
            input_dim: Dimension of encoder features
            hidden_dims: Hidden layer dimensions
            num_classes: Number of language classes
            dropout: Dropout probability
            pooling: Pooling strategy (mean, max, attention)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling = pooling
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # Attention pooling (optional)
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            encoder_hidden_states: Encoder output (batch, seq_len, hidden_dim)
            attention_mask: Optional mask for valid positions
            
        Returns:
            Language logits (batch, num_classes)
        """
        # Pool encoder states
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (encoder_hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                pooled = encoder_hidden_states.mean(dim=1)
                
        elif self.pooling == "max":
            if attention_mask is not None:
                encoder_hidden_states = encoder_hidden_states.masked_fill(
                    ~attention_mask.unsqueeze(-1), float('-inf')
                )
            pooled = encoder_hidden_states.max(dim=1)[0]
            
        elif self.pooling == "attention":
            attn_weights = self.attention(encoder_hidden_states)  # (batch, seq, 1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    ~attention_mask.unsqueeze(-1), float('-inf')
                )
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (encoder_hidden_states * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        # Classify
        logits = self.classifier(pooled)
        return logits
    
    def predict(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict language with probabilities.
        
        Args:
            encoder_hidden_states: Encoder output
            attention_mask: Optional mask
            
        Returns:
            Tuple of (predicted_labels, probabilities)
        """
        logits = self.forward(encoder_hidden_states, attention_mask)
        probs = F.softmax(logits, dim=-1)
        labels = probs.argmax(dim=-1)
        return labels, probs


class AdapterRouter(nn.Module):
    """Routes inputs to appropriate language adapters.
    
    Supports multiple routing strategies:
    - hard: Select single adapter based on argmax
    - soft: Weighted combination of adapter outputs
    - threshold: Use ensemble if uncertain
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        adapters: Dict[str, nn.Module],
        classifier: LanguageClassifier,
        languages: List[str],
        strategy: str = "hard",
        threshold: float = 0.7,
    ):
        """
        Args:
            base_model: Frozen base model (encoder)
            adapters: Dictionary mapping language -> adapter module
            classifier: Language classifier
            languages: Ordered list of language names
            strategy: Routing strategy (hard, soft, threshold)
            threshold: Confidence threshold for threshold strategy
        """
        super().__init__()
        
        self.base_model = base_model
        self.adapters = nn.ModuleDict(adapters)
        self.classifier = classifier
        self.languages = languages
        self.strategy = strategy
        self.threshold = threshold
        
        # Freeze base model and classifier
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.classifier.parameters():
            param.requires_grad = False
            
        # Language to index mapping
        self.lang_to_idx = {lang: i for i, lang in enumerate(languages)}
        
    def extract_encoder_features(
        self,
        input_features: torch.Tensor,
    ) -> torch.Tensor:
        """Extract encoder features from base model.
        
        Args:
            input_features: Log-mel spectrogram
            
        Returns:
            Encoder hidden states
        """
        with torch.no_grad():
            encoder_outputs = self.base_model.model.encoder(input_features)
            return encoder_outputs.last_hidden_state
    
    def detect_language(
        self,
        encoder_hidden_states: torch.Tensor,
    ) -> Tuple[List[str], torch.Tensor]:
        """Detect language from encoder features.
        
        Args:
            encoder_hidden_states: Encoder output
            
        Returns:
            Tuple of (predicted languages, probabilities)
        """
        with torch.no_grad():
            labels, probs = self.classifier.predict(encoder_hidden_states)
            
        predicted_langs = [self.languages[l.item()] for l in labels]
        return predicted_langs, probs
    
    def forward(
        self,
        input_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with adaptive routing.
        
        Args:
            input_features: Log-mel spectrogram
            labels: Optional target labels
            **kwargs: Additional arguments
            
        Returns:
            Model outputs
        """
        # Extract encoder features
        encoder_hidden_states = self.extract_encoder_features(input_features)
        
        # Detect language
        predicted_langs, probs = self.detect_language(encoder_hidden_states)
        
        if self.strategy == "hard":
            return self._hard_routing(input_features, predicted_langs, labels, **kwargs)
        elif self.strategy == "soft":
            return self._soft_routing(input_features, probs, labels, **kwargs)
        elif self.strategy == "threshold":
            return self._threshold_routing(input_features, probs, labels, **kwargs)
        else:
            raise ValueError(f"Unknown routing strategy: {self.strategy}")
    
    def _hard_routing(
        self,
        input_features: torch.Tensor,
        predicted_langs: List[str],
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Hard routing: use single best adapter per sample."""
        batch_size = input_features.shape[0]
        outputs = []
        
        for i in range(batch_size):
            lang = predicted_langs[i]
            adapter = self.adapters[lang]
            
            sample_input = input_features[i:i+1]
            sample_label = labels[i:i+1] if labels is not None else None
            
            output = adapter(
                input_features=sample_input,
                labels=sample_label,
                **kwargs,
            )
            outputs.append(output)
        
        # Aggregate outputs
        return self._aggregate_outputs(outputs)
    
    def _soft_routing(
        self,
        input_features: torch.Tensor,
        probs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Soft routing: weighted combination of all adapters."""
        batch_size = input_features.shape[0]
        
        # Run all adapters
        adapter_outputs = {}
        for lang, adapter in self.adapters.items():
            adapter_outputs[lang] = adapter(
                input_features=input_features,
                labels=labels,
                output_hidden_states=True,
                **kwargs,
            )
        
        # Weight logits by language probabilities
        weighted_logits = None
        for i, lang in enumerate(self.languages):
            lang_prob = probs[:, i:i+1, None]  # (batch, 1, 1)
            logits = adapter_outputs[lang].logits  # (batch, seq, vocab)
            
            if weighted_logits is None:
                weighted_logits = lang_prob * logits
            else:
                weighted_logits = weighted_logits + lang_prob * logits
        
        # Compute weighted loss if labels provided
        loss = None
        if labels is not None:
            loss = sum(
                probs[:, i].mean() * adapter_outputs[lang].loss
                for i, lang in enumerate(self.languages)
            )
        
        return {
            "loss": loss,
            "logits": weighted_logits,
            "probs": probs,
        }
    
    def _threshold_routing(
        self,
        input_features: torch.Tensor,
        probs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Threshold routing: hard if confident, soft top-k if uncertain."""
        max_probs, max_indices = probs.max(dim=-1)
        confident = max_probs > self.threshold
        
        if confident.all():
            # All samples confident, use hard routing
            predicted_langs = [self.languages[i.item()] for i in max_indices]
            return self._hard_routing(input_features, predicted_langs, labels, **kwargs)
        elif (~confident).all():
            # All samples uncertain, use soft routing
            return self._soft_routing(input_features, probs, labels, **kwargs)
        else:
            # Mixed: process separately
            # For simplicity, fall back to soft routing
            return self._soft_routing(input_features, probs, labels, **kwargs)
    
    def _aggregate_outputs(
        self,
        outputs: List[Dict],
    ) -> Dict[str, torch.Tensor]:
        """Aggregate outputs from multiple samples."""
        if not outputs:
            return {}
            
        result = {}
        
        # Stack losses
        if outputs[0].loss is not None:
            result["loss"] = torch.stack([o.loss for o in outputs]).mean()
        
        # Stack logits
        if hasattr(outputs[0], "logits") and outputs[0].logits is not None:
            result["logits"] = torch.cat([o.logits for o in outputs], dim=0)
        
        return result
    
    def generate(
        self,
        input_features: torch.Tensor,
        language: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Generate transcription with routing.
        
        Args:
            input_features: Log-mel spectrogram
            language: Optional language override (skip detection)
            **kwargs: Generation arguments
            
        Returns:
            Generated token IDs
        """
        if language is not None:
            # Use specified language adapter
            adapter = self.adapters[language]
            return adapter.generate(input_features, **kwargs)
        
        # Detect language
        encoder_hidden_states = self.extract_encoder_features(input_features)
        predicted_langs, probs = self.detect_language(encoder_hidden_states)
        
        # Generate with detected language adapter
        batch_size = input_features.shape[0]
        all_generated = []
        
        for i in range(batch_size):
            lang = predicted_langs[i]
            adapter = self.adapters[lang]
            
            sample_input = input_features[i:i+1]
            generated = adapter.generate(sample_input, **kwargs)
            all_generated.append(generated)
        
        # Pad and stack
        max_len = max(g.shape[1] for g in all_generated)
        padded = []
        for g in all_generated:
            if g.shape[1] < max_len:
                pad = torch.zeros(1, max_len - g.shape[1], dtype=g.dtype, device=g.device)
                g = torch.cat([g, pad], dim=1)
            padded.append(g)
        
        return torch.cat(padded, dim=0)
