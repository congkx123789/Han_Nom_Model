from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import os

# Resolve paths relative to this script's location (project root = parent of output/)
PROJECT_ROOT = Path(__file__).parent.parent

print("Loading model...", flush=True)
ckpt_dir = str(PROJECT_ROOT / "output/got_ocr_hannom_swift/v6-20260303-160846/checkpoint-9580")
model_id = "stepfun-ai/GOT-OCR2_0"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
base_model = AutoModel.from_pretrained(model_id, trust_remote_code=True, device_map="cuda", use_safetensors=True)
model = PeftModel.from_pretrained(base_model, ckpt_dir)
model = model.merge_and_unload()
model.eval()

# Fix attention mask / pad_token_id warnings
model.config.pad_token_id = tokenizer.eos_token_id

print("Model loaded!\n", flush=True)

# Nôm test images from Tale of Kiều
nom_tests = [
    ("Tale of Kieu 1866/page065b_5.jpg", "㐌衝身世群算浽芇"),
    ("Tale of Kieu 1872/page75a_9.jpg",  "𢚸貞払𡏡劳刀󰡪蜍"),
    ("Tale of Kieu 1866/page01a_6.jpg",  "稿𦹳吝󰇾𠓀畑"),
    ("Tale of Kieu 1871/page006_19.jpg", "𦹵坡味襖染𡽫䏧𡗶"),
    ("Tale of Kieu 1872/page28a_12.jpg", "𢞂𬂙𩙌捲󰘚溋"),
]

base_dir = str(PROJECT_ROOT / "data/NomNaOCR_dataset/Patches")

for img_rel, gt in nom_tests:
    img_path = os.path.join(base_dir, img_rel)
    abs_path = os.path.abspath(img_path)
    if os.path.exists(abs_path):
        try:
            pred = model.chat(tokenizer, abs_path, ocr_type="ocr")
            match = "✓" if pred.strip() == gt else "✗"
            print(f"{match} {os.path.basename(img_rel)}", flush=True)
            print(f"  Ground Truth: {gt}", flush=True)
            print(f"  Prediction:   {pred.strip()}", flush=True)
            print(flush=True)
        except Exception as e:
            print(f"ERROR on {img_rel}: {e}", flush=True)
    else:
        print(f"NOT FOUND: {abs_path}", flush=True)
