"""ASR metrics computation."""

import logging
from typing import Dict, List, Optional, Tuple

import evaluate

logger = logging.getLogger(__name__)

# Load metrics once
_wer_metric = None
_cer_metric = None


def get_wer_metric():
    """Get WER metric (lazy loading)."""
    global _wer_metric
    if _wer_metric is None:
        _wer_metric = evaluate.load("wer")
    return _wer_metric


def get_cer_metric():
    """Get CER metric (lazy loading)."""
    global _cer_metric
    if _cer_metric is None:
        _cer_metric = evaluate.load("cer")
    return _cer_metric


def compute_wer(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute Word Error Rate.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        WER as a float (0-1 range, can exceed 1)
    """
    if not predictions or not references:
        return 0.0
        
    wer_metric = get_wer_metric()
    
    # Handle empty strings
    predictions = [p if p.strip() else "<empty>" for p in predictions]
    references = [r if r.strip() else "<empty>" for r in references]
    
    return wer_metric.compute(predictions=predictions, references=references)


def compute_cer(
    predictions: List[str],
    references: List[str],
) -> float:
    """Compute Character Error Rate.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        CER as a float (0-1 range, can exceed 1)
    """
    if not predictions or not references:
        return 0.0
        
    cer_metric = get_cer_metric()
    
    # Handle empty strings
    predictions = [p if p.strip() else "<empty>" for p in predictions]
    references = [r if r.strip() else "<empty>" for r in references]
    
    return cer_metric.compute(predictions=predictions, references=references)


def compute_metrics(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    """Compute all ASR metrics.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        Dictionary with WER and CER
    """
    return {
        "wer": compute_wer(predictions, references),
        "cer": compute_cer(predictions, references),
    }


def compute_metrics_per_sample(
    predictions: List[str],
    references: List[str],
) -> List[Dict[str, float]]:
    """Compute metrics per sample.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        
    Returns:
        List of dictionaries with per-sample metrics
    """
    results = []
    
    for pred, ref in zip(predictions, references):
        results.append({
            "wer": compute_wer([pred], [ref]),
            "cer": compute_cer([pred], [ref]),
            "ref_words": len(ref.split()),
            "pred_words": len(pred.split()),
        })
    
    return results


def analyze_errors(
    predictions: List[str],
    references: List[str],
    top_k: int = 20,
) -> Dict:
    """Analyze common errors in predictions.
    
    Args:
        predictions: List of predicted transcriptions
        references: List of reference transcriptions
        top_k: Number of top errors to return
        
    Returns:
        Dictionary with error analysis
    """
    from collections import Counter
    
    substitutions = Counter()
    insertions = Counter()
    deletions = Counter()
    
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        # Simple word-level alignment
        pred_set = set(pred_words)
        ref_set = set(ref_words)
        
        # Words in ref but not in pred (potential deletions)
        for word in ref_set - pred_set:
            deletions[word] += 1
            
        # Words in pred but not in ref (potential insertions)
        for word in pred_set - ref_set:
            insertions[word] += 1
    
    return {
        "top_deletions": deletions.most_common(top_k),
        "top_insertions": insertions.most_common(top_k),
        "total_deletions": sum(deletions.values()),
        "total_insertions": sum(insertions.values()),
    }
