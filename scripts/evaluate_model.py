"""Evaluation script for trained LoRA adapters."""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import get_processor
from src.models.whisper_lora import load_whisper_lora_from_checkpoint
from src.data import create_dataset, DataCollatorSpeechSeq2Seq
from src.evaluation import ASREvaluator

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA adapters")
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="whisper-small",
        choices=["whisper-small", "whisper-medium", "whisper-large"],
        help="Base model",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to adapter checkpoint",
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        choices=["hindi", "italian", "punjabi", "telugu"],
        help="Target language",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_sources",
        type=str,
        nargs="+",
        default=["common_voice"],
        help="Data sources to use",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    # Generation arguments
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum tokens to generate")
    parser.add_argument("--num_beams", type=int, default=1, help="Beam search width")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for results")
    parser.add_argument("--save_predictions", action="store_true", help="Save individual predictions")
    
    # Other arguments
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache directory")
    
    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = load_whisper_lora_from_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        language=args.language,
        device=device,
    )
    
    processor = model.processor
    
    # Load dataset
    logger.info(f"Loading {args.split} dataset for {args.language}")
    
    test_dataset = create_dataset(
        language=args.language,
        split=args.split,
        sources=args.data_sources,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        processor=processor,
    )
    
    logger.info(f"Evaluation samples: {len(test_dataset)}")
    
    # Create data collator and loader
    data_collator = DataCollatorSpeechSeq2Seq(processor=processor)
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create evaluator
    evaluator = ASREvaluator(
        model=model,
        processor=processor,
        device=device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        test_dataloader,
        return_predictions=args.save_predictions,
    )
    
    # Print results
    logger.info("=" * 50)
    logger.info(f"Evaluation Results for {args.language}")
    logger.info("=" * 50)
    logger.info(f"WER: {results['wer']:.4f} ({results['wer']*100:.2f}%)")
    logger.info(f"CER: {results['cer']:.4f} ({results['cer']*100:.2f}%)")
    logger.info(f"Samples: {results['num_samples']}")
    logger.info("=" * 50)
    
    # Save results
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        metrics = {
            "model": args.model,
            "checkpoint": args.checkpoint,
            "language": args.language,
            "split": args.split,
            "wer": results["wer"],
            "cer": results["cer"],
            "num_samples": results["num_samples"],
            "num_beams": args.num_beams,
        }
        
        with open(output_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Saved metrics to {output_dir / 'metrics.json'}")
        
        # Save predictions if requested
        if args.save_predictions and "predictions" in results:
            predictions_file = output_dir / "predictions.txt"
            references_file = output_dir / "references.txt"
            
            with open(predictions_file, "w") as f:
                for pred in results["predictions"]:
                    f.write(pred + "\n")
            
            with open(references_file, "w") as f:
                for ref in results["references"]:
                    f.write(ref + "\n")
            
            logger.info(f"Saved predictions to {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
