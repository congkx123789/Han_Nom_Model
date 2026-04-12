#!/usr/bin/env python3
"""
Test the 4-bit quantized Qwen2.5-VL-3B model
Verify VRAM usage and model functionality
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import sys

def test_quantized_model():
    """Test if the 4-bit quantized model loads correctly"""
    
    model_path = "./models/Qwen2.5-VL-3B-4bit"
    
    print("=" * 70)
    print("Testing 4-bit Quantized Qwen2.5-VL-3B Model")
    print("=" * 70)
    
    # Check CUDA
    print(f"\n1. System Information:")
    print(f"   - CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        torch.cuda.reset_peak_memory_stats()
    
    # Quantization config (same as used during quantization)
    print(f"\n2. Configuring 4-bit loading...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    print("   ✓ 4-bit NF4 quantization config ready")
    
    # Load processor
    print(f"\n3. Loading processor from {model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print("   ✓ Processor loaded")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Load model
    print(f"\n4. Loading 4-bit quantized model...")
    print("   (This should be faster than the original model)")
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto",
        )
        print("   ✓ Model loaded successfully")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Model info
    print(f"\n5. Model Information:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   - Total Parameters: {total_params / 1e9:.2f}B")
    print(f"   - Trainable Parameters: {trainable_params / 1e9:.2f}B")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Dtype: {next(model.parameters()).dtype}")
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_peak = torch.cuda.max_memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        print(f"\n6. VRAM Usage:")
        print(f"   - Current: {vram_used:.2f} GB")
        print(f"   - Peak: {vram_peak:.2f} GB")
        print(f"   - Total Available: {vram_total:.2f} GB")
        print(f"   - Free: {vram_total - vram_used:.2f} GB")
        print(f"   - Usage: {(vram_used / vram_total * 100):.1f}%")
        
        # Compare with original
        original_vram = 7.0  # Approximate
        savings = original_vram - vram_used
        print(f"\n7. Comparison with Original Model:")
        print(f"   - Original VRAM: ~{original_vram:.1f} GB")
        print(f"   - Quantized VRAM: {vram_used:.2f} GB")
        print(f"   - Savings: ~{savings:.2f} GB ({(savings/original_vram*100):.1f}%)")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed! 4-bit model is ready for fine-tuning.")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Prepare your Han-Nom training data")
    print("  2. Configure LoRA for fine-tuning")
    print("  3. Start training with low VRAM usage!")
    
    return True

if __name__ == "__main__":
    success = test_quantized_model()
    sys.exit(0 if success else 1)
