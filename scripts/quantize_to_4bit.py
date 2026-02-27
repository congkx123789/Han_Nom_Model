#!/usr/bin/env python3
"""
Quantize Qwen2.5-VL-3B to 4-bit using QLoRA
This reduces VRAM usage from ~7GB to ~2.5-3GB
"""

import torch
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig
)
from peft import prepare_model_for_kbit_training
import os

def quantize_model():
    """Load and quantize the model to 4-bit"""
    
    model_path = "./models/Qwen2.5-VL-3B"
    output_path = "./models/Qwen2.5-VL-3B-4bit"
    
    print("=" * 70)
    print("Quantizing Qwen2.5-VL-3B to 4-bit QLoRA")
    print("=" * 70)
    
    # Check CUDA
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available. This script requires a GPU.")
        print("   Continuing anyway, but model will load on CPU (very slow).")
    else:
        print(f"\n✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    # 4-bit quantization config
    print("\n1. Configuring 4-bit quantization...")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,                      # Enable 4-bit loading
        bnb_4bit_compute_dtype=torch.float16,   # Compute in fp16
        bnb_4bit_quant_type="nf4",              # Use NormalFloat4 quantization
        bnb_4bit_use_double_quant=True,         # Double quantization for extra compression
    )
    print("   ✓ Config: NF4 quantization with double quant")
    print("   ✓ Compute dtype: float16")
    
    # Load processor
    print(f"\n2. Loading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    print("   ✓ Processor loaded")
    
    # Load model with quantization
    print(f"\n3. Loading model with 4-bit quantization...")
    print("   (This will take a few minutes...)")
    
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        quantization_config=quantization_config,
        device_map="auto",
    )
    print("   ✓ Model loaded and quantized to 4-bit")
    
    # Prepare for QLoRA training
    print("\n4. Preparing model for QLoRA training...")
    model = prepare_model_for_kbit_training(model)
    print("   ✓ Model prepared for k-bit training")
    
    # Model info
    print(f"\n5. Model Information:")
    print(f"   - Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"   - Device: {next(model.parameters()).device}")
    print(f"   - Dtype: {next(model.parameters()).dtype}")
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"   - VRAM Usage: {vram_used:.2f} GB / {vram_total:.2f} GB")
        print(f"   - VRAM Saved: ~{7 - vram_used:.2f} GB (compared to full precision)")
    
    # Save quantized model
    print(f"\n6. Saving quantized model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    
    # Save model
    model.save_pretrained(output_path)
    processor.save_pretrained(output_path)
    
    print("   ✓ Model saved")
    
    # Save config info
    config_info = f"""# Qwen2.5-VL-3B 4-bit Quantized Model

## Quantization Details
- **Method**: 4-bit QLoRA (NormalFloat4)
- **Double Quantization**: Enabled
- **Compute Dtype**: float16
- **VRAM Usage**: ~2.5-3GB (vs ~7GB full precision)

## Usage

```python
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "./models/Qwen2.5-VL-3B-4bit",
    quantization_config=quantization_config,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("./models/Qwen2.5-VL-3B-4bit")
```

## Original Model
- Location: `../Qwen2.5-VL-3B/`
- Size: 7.1GB
"""
    
    with open(os.path.join(output_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(config_info)
    
    print("\n" + "=" * 70)
    print("✓ Quantization Complete!")
    print("=" * 70)
    print(f"\nQuantized model saved to: {output_path}")
    print(f"VRAM savings: ~{7 - (vram_used if torch.cuda.is_available() else 0):.1f} GB")
    print("\nYou can now use this model for fine-tuning with much lower VRAM!")
    
    return model, processor

if __name__ == "__main__":
    model, processor = quantize_model()
