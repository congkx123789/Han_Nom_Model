"""
Download GOT-OCR-2.0 model from Hugging Face
"""

from transformers import AutoModel, AutoTokenizer
import torch

# Model configuration
model_name = "stepfun-ai/GOT-OCR2_0"
cache_dir = "./models/GOT_OCR_2_0"

print(f"Downloading {model_name}...")
print(f"This may take a few minutes (~580MB download)")

# 1. Download Tokenizer
print("\n[1/2] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    model_name, 
    trust_remote_code=True,
    cache_dir=cache_dir
)

# 2. Download Model (FP16 for efficiency on RTX 5060 Ti)
print("\n[2/2] Downloading model...")
model = AutoModel.from_pretrained(
    model_name,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    device_map='cuda',
    use_safetensors=True,
    pad_token_id=tokenizer.eos_token_id,
    cache_dir=cache_dir
)

# Set to evaluation mode
model = model.eval().cuda()

print(f"\n✅ Download complete! Model saved to: {cache_dir}")
print(f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
print(f"GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
