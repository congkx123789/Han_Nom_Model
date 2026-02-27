"""
Test GOT-OCR-2.0 on Han Nom images
"""

import torch
from transformers import AutoModel, AutoTokenizer
import os
import glob
import argparse

# Configuration
MODEL_PATH = "stepfun-ai/GOT-OCR2_0"
CROPS_DIR = "data/yolo_qwen_ocr_results"

def load_model():
    print(f"Loading GOT-OCR from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        device_map='cuda', 
        use_safetensors=True
    )
    model = model.eval().cuda()
    return model, tokenizer

def run_ocr(model, tokenizer, image_path):
    """Run OCR on a single image"""
    try:
        # GOT-OCR has a simple .chat() interface
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
        print(f"\nTesting on: {args.image}")
        result = run_ocr(model, tokenizer, args.image)
        print(f"\nResult:\n{result}")
    
    elif args.test_all:
        # Test on all crops
        crop_files = sorted(glob.glob(os.path.join(CROPS_DIR, '*.jpg')))[:10]
        print(f"\nTesting on {len(crop_files)} images...")
        
        for crop_path in crop_files:
            crop_name = os.path.basename(crop_path)
            result = run_ocr(model, tokenizer, crop_path)
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
                result = run_ocr(model, tokenizer, img_path)
                print(f"\n{img_name}:")
                print(f"  {result}")
            else:
                print(f"\n{img_name}: File not found")

if __name__ == "__main__":
    main()
