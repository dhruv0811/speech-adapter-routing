"""Adapter routing for language detection and selection."""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class LanguageClassifier(nn.Module):
    """Lightweight language classifier for adapter routing (LID).
    
    Takes encoder features and predicts language probabilities.
    Supports multiple architectures:
    - MLP: Simple feed-forward on pooled features
    - CNN: 1D convolution for temporal modeling before pooling
    - Attention: Self-attention pooling with learnable query
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 4,
        dropout: float = 0.3,
        pooling: str = "mean",
        use_layer_norm: bool = True,
        use_cnn: bool = False,
        cnn_channels: int = 256,
        cnn_kernel_size: int = 5,
        label_smoothing: float = 0.0,
        languages: Optional[List[str]] = None,
    ):
        """
        Args:
            input_dim: Dimension of encoder features (768 for Whisper Small/Medium, 1280 for Large)
            hidden_dims: Hidden layer dimensions for MLP classifier
            num_classes: Number of language classes
            dropout: Dropout probability
            pooling: Pooling strategy (mean, max, attention)
            use_layer_norm: Apply layer normalization to input features
            use_cnn: Use 1D CNN for temporal modeling before pooling
            cnn_channels: Number of CNN output channels
            cnn_kernel_size: CNN kernel size
            label_smoothing: Label smoothing for cross-entropy loss
            languages: Ordered list of language names for index mapping
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.pooling = pooling
        self.use_cnn = use_cnn
        self.label_smoothing = label_smoothing
        self.languages = languages or [f"lang_{i}" for i in range(num_classes)]
        
        # Language to index mapping
        self.lang_to_idx = {lang: i for i, lang in enumerate(self.languages)}
        self.idx_to_lang = {i: lang for i, lang in enumerate(self.languages)}
        
        # Optional layer normalization on input
        self.layer_norm = nn.LayerNorm(input_dim) if use_layer_norm else nn.Identity()
        
        # Optional 1D CNN for temporal modeling
        classifier_input_dim = input_dim
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv1d(input_dim, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Conv1d(cnn_channels, cnn_channels, kernel_size=cnn_kernel_size, padding=cnn_kernel_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_input_dim = cnn_channels
        
        # Build MLP classifier layers
        layers = []
        prev_dim = classifier_input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
        # Attention pooling (optional)
        if pooling == "attention":
            self.attention = nn.Sequential(
                nn.Linear(classifier_input_dim, 128),
                nn.Tanh(),
                nn.Linear(128, 1),
            )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        
    def _pool_features(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pool sequence features to single vector.
        
        Args:
            features: Sequence features (batch, seq_len, hidden_dim)
            attention_mask: Optional mask for valid positions
            
        Returns:
            Pooled features (batch, hidden_dim)
        """
        if self.pooling == "mean":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                pooled = (features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
            else:
                pooled = features.mean(dim=1)
                
        elif self.pooling == "max":
            if attention_mask is not None:
                features = features.masked_fill(
                    ~attention_mask.unsqueeze(-1), float('-inf')
                )
            pooled = features.max(dim=1)[0]
            
        elif self.pooling == "attention":
            attn_weights = self.attention(features)  # (batch, seq, 1)
            if attention_mask is not None:
                attn_weights = attn_weights.masked_fill(
                    ~attention_mask.unsqueeze(-1), float('-inf')
                )
            attn_weights = F.softmax(attn_weights, dim=1)
            pooled = (features * attn_weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return pooled
        
    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            encoder_hidden_states: Encoder output (batch, seq_len, hidden_dim)
            attention_mask: Optional mask for valid positions
            labels: Optional language labels for computing loss (batch,)
            
        Returns:
            Dictionary with logits, (optional) loss, and probabilities
        """
        # Apply layer normalization
        features = self.layer_norm(encoder_hidden_states)
        
        # Optional CNN for temporal modeling
        if self.use_cnn:
            # CNN expects (batch, channels, seq_len)
            features = features.transpose(1, 2)
            features = self.cnn(features)
            features = features.transpose(1, 2)
        
        # Pool encoder states
        pooled = self._pool_features(features, attention_mask)
        
        # Classify
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        
        return {
            "logits": logits,
            "probs": probs,
            "loss": loss,
        }
    
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
        outputs = self.forward(encoder_hidden_states, attention_mask)
        probs = outputs["probs"]
        labels = probs.argmax(dim=-1)
        return labels, probs
    
    def predict_language(
        self,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[List[str], torch.Tensor]:
        """Predict language names with probabilities.
        
        Args:
            encoder_hidden_states: Encoder output
            attention_mask: Optional mask
            
        Returns:
            Tuple of (predicted_language_names, probabilities)
        """
        labels, probs = self.predict(encoder_hidden_states, attention_mask)
        lang_names = [self.idx_to_lang[l.item()] for l in labels]
        return lang_names, probs
    
    def save(self, save_path: Union[str, Path]):
        """Save classifier checkpoint.
        
        Args:
            save_path: Path to save checkpoint
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
                "pooling": self.pooling,
                "use_cnn": self.use_cnn,
                "label_smoothing": self.label_smoothing,
                "languages": self.languages,
            }
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved classifier to {save_path}")
    
    @classmethod
    def load(cls, load_path: Union[str, Path], device: Optional[str] = None) -> "LanguageClassifier":
        """Load classifier from checkpoint.
        
        Args:
            load_path: Path to checkpoint
            device: Device to load on
            
        Returns:
            Loaded LanguageClassifier
        """
        load_path = Path(load_path)
        checkpoint = torch.load(load_path, map_location=device or "cpu")
        
        # Reconstruct config - handle both old and new checkpoint formats
        config = checkpoint.get("config", {})
        
        classifier = cls(
            input_dim=config.get("input_dim", 768),
            num_classes=config.get("num_classes", 4),
            pooling=config.get("pooling", "mean"),
            use_cnn=config.get("use_cnn", False),
            label_smoothing=config.get("label_smoothing", 0.0),
            languages=config.get("languages"),
        )
        
        classifier.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Loaded classifier from {load_path}")
        
        return classifier


class EncoderFeatureExtractor(nn.Module):
    """Helper class to extract encoder features from Whisper models.
    
    Handles different model wrapping structures (raw, PEFT, etc.)
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_index: int = -1,
    ):
        """
        Args:
            model: Base model or wrapped model (WhisperLoRA, etc.)
            layer_index: Which encoder layer to extract from (-1 = last layer)
        """
        super().__init__()
        self.model = model
        self.layer_index = layer_index
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _get_encoder(self) -> nn.Module:
        """Get encoder from wrapped model."""
        model = self.model
        
        # Handle different wrapping structures
        # WhisperLoRA: model.model.base_model.model.model.encoder
        # Raw Whisper: model.model.encoder
        # PEFT Whisper: model.base_model.model.model.encoder
        
        if hasattr(model, 'model'):
            model = model.model
        
        if hasattr(model, 'base_model'):
            model = model.base_model
            if hasattr(model, 'model'):
                model = model.model
        
        if hasattr(model, 'model'):
            model = model.model
            
        if hasattr(model, 'encoder'):
            return model.encoder
        
        raise ValueError(f"Could not find encoder in model structure: {type(self.model)}")
    
    @torch.no_grad()
    def forward(
        self,
        input_features: torch.Tensor,
        output_hidden_states: bool = True,
    ) -> torch.Tensor:
        """Extract encoder features.
        
        Args:
            input_features: Log-mel spectrogram (batch, n_mels, time)
            output_hidden_states: Whether to return all hidden states
            
        Returns:
            Encoder hidden states from specified layer
        """
        encoder = self._get_encoder()
        
        # Get all hidden states if we need a specific layer
        encoder_outputs = encoder(
            input_features,
            output_hidden_states=output_hidden_states or (self.layer_index != -1),
            return_dict=True,
        )
        
        if self.layer_index == -1:
            # Last layer
            return encoder_outputs.last_hidden_state
        else:
            # Specific layer
            if hasattr(encoder_outputs, 'hidden_states') and encoder_outputs.hidden_states is not None:
                return encoder_outputs.hidden_states[self.layer_index]
            else:
                logger.warning(f"Hidden states not available, using last hidden state")
                return encoder_outputs.last_hidden_state
    
    def get_hidden_dim(self) -> int:
        """Get hidden dimension of encoder."""
        encoder = self._get_encoder()
        if hasattr(encoder, 'config'):
            return encoder.config.d_model
        # Fallback - try to infer from layer
        for name, module in encoder.named_modules():
            if isinstance(module, nn.Linear):
                return module.out_features
        return 768  # Default for Whisper Small/Medium


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
        
        # Create feature extractor for the base model
        self.feature_extractor = EncoderFeatureExtractor(base_model)
        
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
        return self.feature_extractor(input_features)
    
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
