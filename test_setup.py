"""Quick test script to verify installation and model loading."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.data import ASRDataset, DataCollatorSpeechSeq2Seq, AudioPreprocessor
        print("  ✓ src.data")
    except ImportError as e:
        print(f"  ✗ src.data: {e}")
        return False
    
    try:
        from src.models import WhisperLoRA, get_processor
        print("  ✓ src.models")
    except ImportError as e:
        print(f"  ✗ src.models: {e}")
        return False
    
    try:
        from src.training import ASRTrainer, compute_wer, compute_cer
        print("  ✓ src.training")
    except ImportError as e:
        print(f"  ✗ src.training: {e}")
        return False
    
    try:
        from src.evaluation import ASREvaluator
        print("  ✓ src.evaluation")
    except ImportError as e:
        print(f"  ✗ src.evaluation: {e}")
        return False
    
    return True


def test_model_loading():
    """Test that models can be loaded."""
    print("\nTesting model loading...")
    
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Device: {device}")
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False
    
    try:
        from src.models import get_processor
        processor = get_processor("whisper-small", language="hindi")
        print("  ✓ Processor loaded")
    except Exception as e:
        print(f"  ✗ Processor: {e}")
        return False
    
    # Only test full model loading if GPU available
    if device == "cuda":
        try:
            from src.models import WhisperLoRA
            print("  Loading WhisperLoRA (this may take a moment)...")
            model = WhisperLoRA(
                model_name="whisper-small",
                lora_r=8,
                language="hindi",
                device=device,
            )
            print("  ✓ WhisperLoRA loaded")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  ✗ WhisperLoRA: {e}")
            return False
    else:
        print("  ⚠ Skipping model loading test (no GPU)")
    
    return True


def test_metrics():
    """Test metric computation."""
    print("\nTesting metrics...")
    
    try:
        from src.training import compute_wer, compute_cer
        
        predictions = ["hello world", "this is a test"]
        references = ["hello world", "this is a test"]
        
        wer = compute_wer(predictions, references)
        cer = compute_cer(predictions, references)
        
        assert wer == 0.0, f"Expected WER=0, got {wer}"
        assert cer == 0.0, f"Expected CER=0, got {cer}"
        
        print(f"  ✓ Perfect match: WER={wer}, CER={cer}")
        
        # Test with errors
        predictions = ["hello word", "this is test"]
        wer = compute_wer(predictions, references)
        cer = compute_cer(predictions, references)
        
        assert wer > 0, "Expected WER > 0"
        print(f"  ✓ With errors: WER={wer:.4f}, CER={cer:.4f}")
        
    except Exception as e:
        print(f"  ✗ Metrics: {e}")
        return False
    
    return True


def test_dataset_config():
    """Test dataset configuration loading."""
    print("\nTesting configs...")
    
    try:
        import yaml
        
        config_files = [
            "configs/model_configs/whisper.yaml",
            "configs/lora_configs/default.yaml",
            "configs/training_configs/default.yaml",
            "configs/dataset_configs/default.yaml",
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            if path.exists():
                with open(path) as f:
                    config = yaml.safe_load(f)
                print(f"  ✓ {config_file}")
            else:
                print(f"  ✗ {config_file} not found")
                return False
    except Exception as e:
        print(f"  ✗ Config loading: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("Speech Adapter Routing - Installation Test")
    print("=" * 50)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_metrics()
    all_passed &= test_dataset_config()
    all_passed &= test_model_loading()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("All tests passed! ✓")
        print("\nNext steps:")
        print("  1. Test data loading:")
        print("     python -c \"from src.data import load_fleurs; ds = load_fleurs('hi_in', 'test'); print(len(ds))\"")
        print("  2. Run a quick training test:")
        print("     python scripts/train_lora.py --model whisper-small --language hindi --max_steps 10 --output_dir checkpoints/test --no_wandb")
    else:
        print("Some tests failed! ✗")
        print("Please check the errors above.")
    print("=" * 50)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
