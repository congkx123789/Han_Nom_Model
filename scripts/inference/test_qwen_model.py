#!/usr/bin/env python3
"""
Test script to verify Qwen2.5-VL-3B model loading
"""

import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import sys

def test_model_loading():
    """Test if the model can be loaded successfully"""
    
    model_path = "./models/Qwen2.5-VL-3B"
    
    print("=" * 60)
    print("Testing Qwen2.5-VL-3B Model Loading")
    print("=" * 60)
    
    # Check CUDA availability
    print(f"\n1. CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # Load processor
    print(f"\n2. Loading processor from {model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("   ✓ Processor loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load processor: {e}")
        return False
    
    # Load model
    print(f"\n3. Loading model from {model_path}...")
    print("   (This may take a few minutes...)")
    try:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map="auto"
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False
    
    # Model info
    print(f"\n4. Model Information:")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Dtype: {next(model.parameters()).dtype}")
    
    if torch.cuda.is_available():
        print(f"   - VRAM Usage: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 60)
    print("✓ All tests passed! Model is ready for fine-tuning.")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
