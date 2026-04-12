"""
Test GOT-OCR-2.0 on Han Nom images natively with merged weights
"""
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel
import torch
import os
import glob
import argparse

# Resolve paths relative to this script's location (project root = parent of scripts/)
PROJECT_ROOT = Path(__file__).parent.parent

# Configuration
MODEL_ID = "stepfun-ai/GOT-OCR2_0"
CKPT_DIR = str(PROJECT_ROOT / "output/got_ocr_hannom_swift/v6-20260303-160846/checkpoint-9580")
CROPS_DIR = str(PROJECT_ROOT / "data/yolo_qwen_ocr_results")

def load_model():
    print(f"Loading GOT-OCR from {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    print("Loading base model...")
    base_model = AutoModel.from_pretrained(
        MODEL_ID, 
        trust_remote_code=True, 
        device_map='cuda', 
        use_safetensors=True
    )
    
    if os.path.exists(CKPT_DIR):
        print(f"Loading LoRA weights from {CKPT_DIR}...")
        model = PeftModel.from_pretrained(base_model, CKPT_DIR)
        print("Merging weights natively...")
        model = model.merge_and_unload()
    else:
        print(f"Warning: Checkpoint {CKPT_DIR} not found. Running base model.")
        model = base_model
        
    model.eval()
    # Fix attention mask / pad_token_id warnings
    model.config.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def run_ocr(model, tokenizer, image_path):
    """Run OCR on a single image natively"""
    try:
        result = model.chat(tokenizer, image_path, ocr_type='ocr')
        return result.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Single image path to test')
    parser.add_argument('--test-all', action='store_true', help='Test on all crop images')
    args = parser.parse_args()
    
    model, tokenizer = load_model()
    
    if args.image:
        # Test single image
        img_path = os.path.abspath(args.image)
        print(f"\nTesting on: {img_path}")
        result = run_ocr(model, tokenizer, img_path)
        print(f"\nResult:\n{result}")
    
    elif args.test_all:
        # Test on all crops
        crop_files = sorted(glob.glob(os.path.join(CROPS_DIR, '*.jpg')))[:10]
        print(f"\nTesting on {len(crop_files)} images...")
        
        for crop_path in crop_files:
            crop_name = os.path.basename(crop_path)
            abs_path = os.path.abspath(crop_path)
            result = run_ocr(model, tokenizer, abs_path)
            print(f"\n{crop_name}:")
            print(f"  {result[:100]}...")
    
    else:
        # Default: test on a few sample images
        test_images = [
            '1264_199_crop_0.jpg',
            '1264_199_crop_1.jpg', 
            '1264_199_crop_2.jpg',
        ]
        
        print(f"\nTesting on {len(test_images)} sample images...")
        for img_name in test_images:
            img_path = os.path.join(CROPS_DIR, img_name)
            if os.path.exists(img_path):
                result = run_ocr(model, tokenizer, os.path.abspath(img_path))
                print(f"\n{img_name}:")
                print(f"  {result}")
            else:
                print(f"\n{img_name}: File not found")

if __name__ == "__main__":
    main()
