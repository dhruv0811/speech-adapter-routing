#!/usr/bin/env python
"""Test script to verify Common Voice 17.0 loading from HuggingFace."""

from src.data.dataset import load_common_voice

def main():
    print("Testing Common Voice 17.0 from HuggingFace (fixie-ai/common_voice_17_0)...")
    print()
    
    # Test streaming mode first (faster, no full download)
    print("1. Testing Hindi (hi) in streaming mode...")
    ds = load_common_voice("hi", split="train", streaming=True)
    print(f"   Dataset features: {list(ds.features.keys())}")
    sample = next(iter(ds))
    print(f"   Sample keys: {list(sample.keys())}")
    text = sample.get("text", sample.get("sentence", "N/A"))
    print(f"   Text sample: {text[:80]}..." if len(text) > 80 else f"   Text: {text}")
    print()
    
    # Test Italian
    print("2. Testing Italian (it) in streaming mode...")
    ds_it = load_common_voice("it", split="train", streaming=True)
    sample_it = next(iter(ds_it))
    text_it = sample_it.get("text", sample_it.get("sentence", "N/A"))
    print(f"   Text sample: {text_it[:80]}..." if len(text_it) > 80 else f"   Text: {text_it}")
    print()
    
    # Test Telugu
    print("3. Testing Telugu (te) in streaming mode...")
    ds_te = load_common_voice("te", split="train", streaming=True)
    sample_te = next(iter(ds_te))
    text_te = sample_te.get("text", sample_te.get("sentence", "N/A"))
    print(f"   Text sample: {text_te[:80]}..." if len(text_te) > 80 else f"   Text: {text_te}")
    print()
    
    # Test Punjabi
    print("4. Testing Punjabi (pa-IN) in streaming mode...")
    ds_pa = load_common_voice("pa-IN", split="train", streaming=True)
    sample_pa = next(iter(ds_pa))
    text_pa = sample_pa.get("text", sample_pa.get("sentence", "N/A"))
    print(f"   Text sample: {text_pa[:80]}..." if len(text_pa) > 80 else f"   Text: {text_pa}")
    print()
    
    print("SUCCESS: All Common Voice 17.0 languages load correctly!")
    print()
    print("Supported language codes: hi, it, te, pa-IN")

if __name__ == "__main__":
    main()
