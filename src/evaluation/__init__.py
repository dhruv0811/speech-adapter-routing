"""Evaluation utilities for ASR models."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..training.metrics import compute_metrics, compute_metrics_per_sample, analyze_errors

logger = logging.getLogger(__name__)


class ASREvaluator:
    """Evaluator for ASR models."""
    
    def __init__(
        self,
        model,
        processor,
        device: Optional[str] = None,
        max_new_tokens: int = 256,
        num_beams: int = 1,
    ):
        """
        Args:
            model: ASR model
            processor: Whisper processor
            device: Device to evaluate on
            max_new_tokens: Maximum tokens to generate
            num_beams: Beam search width
        """
        self.model = model
        self.processor = processor
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        
        self.model.to(self.device)
        self.model.eval()
        
    def evaluate(
        self,
        dataloader: DataLoader,
        return_predictions: bool = False,
    ) -> Dict:
        """Evaluate model on dataset.
        
        Args:
            dataloader: Data loader for evaluation
            return_predictions: Whether to return individual predictions
            
        Returns:
            Dictionary with evaluation results
        """
        all_predictions = []
        all_references = []
        all_texts = []  # Original texts if available
        
        logger.info("Running evaluation...")
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_features = batch["input_features"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                # Generate predictions
                generated_ids = self.model.generate(
                    input_features=input_features,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
                
                # Decode predictions
                predictions = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                
                # Decode references
                labels = labels.clone()
                labels[labels == -100] = self.processor.tokenizer.pad_token_id
                references = self.processor.batch_decode(
                    labels,
                    skip_special_tokens=True,
                )
                
                all_predictions.extend([p.strip() for p in predictions])
                all_references.extend([r.strip() for r in references])
                
                # Keep original texts if available
                if "text" in batch:
                    all_texts.extend(batch["text"])
        
        # Compute metrics
        metrics = compute_metrics(all_predictions, all_references)
        
        result = {
            "wer": metrics["wer"],
            "cer": metrics["cer"],
            "num_samples": len(all_predictions),
        }
        
        if return_predictions:
            result["predictions"] = all_predictions
            result["references"] = all_references
            if all_texts:
                result["original_texts"] = all_texts
        
        logger.info(f"Evaluation results: WER={result['wer']:.4f}, CER={result['cer']:.4f}")
        
        return result
    
    def evaluate_per_sample(
        self,
        dataloader: DataLoader,
    ) -> List[Dict]:
        """Evaluate model and return per-sample metrics.
        
        Args:
            dataloader: Data loader for evaluation
            
        Returns:
            List of per-sample results
        """
        results = self.evaluate(dataloader, return_predictions=True)
        
        per_sample = compute_metrics_per_sample(
            results["predictions"],
            results["references"],
        )
        
        # Add predictions and references
        for i, sample in enumerate(per_sample):
            sample["prediction"] = results["predictions"][i]
            sample["reference"] = results["references"][i]
        
        return per_sample
    
    def analyze(
        self,
        dataloader: DataLoader,
        top_k: int = 20,
    ) -> Dict:
        """Run full error analysis.
        
        Args:
            dataloader: Data loader for evaluation
            top_k: Number of top errors to analyze
            
        Returns:
            Error analysis results
        """
        results = self.evaluate(dataloader, return_predictions=True)
        
        error_analysis = analyze_errors(
            results["predictions"],
            results["references"],
            top_k=top_k,
        )
        
        # Add overall metrics
        error_analysis["wer"] = results["wer"]
        error_analysis["cer"] = results["cer"]
        error_analysis["num_samples"] = results["num_samples"]
        
        return error_analysis


def transcribe_audio(
    model,
    processor,
    audio_path: Union[str, Path],
    device: Optional[str] = None,
    max_new_tokens: int = 256,
) -> str:
    """Transcribe a single audio file.
    
    Args:
        model: ASR model
        processor: Whisper processor
        audio_path: Path to audio file
        device: Device to use
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        Transcription string
    """
    from ..data.preprocessing import load_audio
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Load audio
    audio, sr = load_audio(audio_path)
    
    # Process
    input_features = processor.feature_extractor(
        audio,
        sampling_rate=sr,
        return_tensors="pt",
    ).input_features.to(device)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_features=input_features,
            max_new_tokens=max_new_tokens,
        )
    
    # Decode
    transcription = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )[0]
    
    return transcription.strip()


def batch_transcribe(
    model,
    processor,
    audio_paths: List[Union[str, Path]],
    batch_size: int = 8,
    device: Optional[str] = None,
    max_new_tokens: int = 256,
) -> List[str]:
    """Transcribe multiple audio files.
    
    Args:
        model: ASR model
        processor: Whisper processor
        audio_paths: List of paths to audio files
        batch_size: Batch size for inference
        device: Device to use
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        List of transcriptions
    """
    from ..data.preprocessing import load_audio
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_transcriptions = []
    
    # Process in batches
    for i in tqdm(range(0, len(audio_paths), batch_size), desc="Transcribing"):
        batch_paths = audio_paths[i:i+batch_size]
        
        # Load and process batch
        input_features_list = []
        for path in batch_paths:
            audio, sr = load_audio(path)
            features = processor.feature_extractor(
                audio,
                sampling_rate=sr,
                return_tensors="pt",
            ).input_features
            input_features_list.append(features)
        
        # Pad to same length
        max_len = max(f.shape[-1] for f in input_features_list)
        padded = []
        for f in input_features_list:
            if f.shape[-1] < max_len:
                pad = torch.zeros(1, f.shape[1], max_len - f.shape[-1])
                f = torch.cat([f, pad], dim=-1)
            padded.append(f)
        
        input_features = torch.cat(padded, dim=0).to(device)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=max_new_tokens,
            )
        
        # Decode
        transcriptions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        all_transcriptions.extend([t.strip() for t in transcriptions])
    
    return all_transcriptions
