"""
Download TrOCR Base model from Hugging Face
"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Model configuration
# Using stage1 (best for fine-tuning from scratch for Han Nom)
model_name = "microsoft/trocr-base-stage1"
cache_dir = "./models/TrOCR_Base"

print(f"Downloading {model_name}...")
print(f"This model is designed for fine-tuning on custom datasets")

# 1. Download Processor (Image processing + Tokenizer)
print("\n[1/2] Downloading processor...")
processor = TrOCRProcessor.from_pretrained(model_name, cache_dir=cache_dir)

# 2. Download Model
print("\n[2/2] Downloading model...")
model = VisionEncoderDecoderModel.from_pretrained(
    model_name,
    cache_dir=cache_dir
)

model.to("cuda")

print(f"\n✅ Download complete! Model saved to: {cache_dir}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"\nNote: This is a base model. You need to fine-tune it on Han Nom data before use.")
