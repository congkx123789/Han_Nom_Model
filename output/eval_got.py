from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch

# Resolve paths relative to this script's location (project root = parent of output/)
PROJECT_ROOT = Path(__file__).parent.parent
CKPT_DIR = str(PROJECT_ROOT / "output/got_ocr_hannom_swift/v6-20260303-160846/checkpoint-9580")
IMG_PATH = str(PROJECT_ROOT / "data/yolo_qwen_ocr_results/1264_199_crop_0.jpg")

model_id = "stepfun-ai/GOT-OCR2_0"

print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

print("Loading base model...", flush=True)
base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device_map="cuda", use_safetensors=True)

print("Loading LoRA adapter...", flush=True)
model = PeftModel.from_pretrained(base_model, CKPT_DIR)

print("Merging weights...", flush=True)
model = model.merge_and_unload()
model.eval()

# Fix attention mask / pad_token_id warnings
model.config.pad_token_id = tokenizer.eos_token_id

print(f"Running inference on {IMG_PATH}...", flush=True)
res = model.chat(tokenizer, IMG_PATH, ocr_type="ocr")
print(f"Result: {res}", flush=True)
