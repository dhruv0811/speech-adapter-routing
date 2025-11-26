#!/usr/bin/env python3
"""Pre-download all datasets for LoRA training.

This script downloads and caches all datasets before training jobs start,
avoiding race conditions and redundant downloads when running array jobs.

Usage:
    python scripts/download_datasets.py
    
Or via UV:
    uv run python scripts/download_datasets.py
"""

import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def download_common_voice(languages: list[str], splits: list[str] = ["train", "validation"]):
    """Download Common Voice datasets for all languages."""
    from src.data.dataset import load_common_voice
    
    for lang in languages:
        for split in splits:
            logger.info(f"Downloading Common Voice: {lang} ({split})")
            try:
                ds = load_common_voice(lang, split=split)
                logger.info(f"  ✓ {len(ds)} samples")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")


def download_ai4bharat(languages: list[str], splits: list[str] = ["train", "valid"]):
    """Download AI4Bharat IndicVoices datasets for Indic languages."""
    from src.data.dataset import load_ai4bharat
    
    for lang in languages:
        for split in splits:
            logger.info(f"Downloading AI4Bharat: {lang} ({split})")
            try:
                ds = load_ai4bharat(lang, split=split)
                logger.info(f"  ✓ {len(ds)} samples")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")


def download_mls(languages: list[str], splits: list[str] = ["train", "validation"]):
    """Download Multilingual LibriSpeech datasets."""
    from src.data.dataset import load_mls
    
    for lang in languages:
        for split in splits:
            logger.info(f"Downloading MLS: {lang} ({split})")
            try:
                ds = load_mls(lang, split=split)
                logger.info(f"  ✓ {len(ds)} samples")
            except Exception as e:
                logger.error(f"  ✗ Failed: {e}")


def main():
    """Download all datasets used in training."""
    
    # Set cache directory
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    logger.info(f"Cache directory: {cache_dir}")
    
    print("\n" + "="*60)
    print("Downloading all datasets for LoRA training")
    print("="*60 + "\n")
    
    # Languages used in training
    all_languages = ["hindi", "italian", "punjabi", "telugu"]
    indic_languages = ["hindi", "punjabi", "telugu"]
    
    # Download Common Voice for all languages
    print("\n[1/3] Common Voice (all languages)")
    print("-" * 40)
    download_common_voice(all_languages)
    
    # Download AI4Bharat for Indic languages
    print("\n[2/3] AI4Bharat IndicVoices (Indic languages)")
    print("-" * 40)
    download_ai4bharat(indic_languages)
    
    # Download MLS for Italian
    print("\n[3/3] Multilingual LibriSpeech (Italian)")
    print("-" * 40)
    download_mls(["italian"])
    
    print("\n" + "="*60)
    print("Dataset download complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
