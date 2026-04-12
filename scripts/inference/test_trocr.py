"""
Test TrOCR Base model on Han Nom images
Note: This is an untrained base model, so results will be poor/random
"""

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch
import os
import glob

# Configuration
MODEL_PATH = "microsoft/trocr-base-stage1"
CROPS_DIR = "data/yolo_qwen_ocr_results"

def load_model():
    print(f"Loading TrOCR from {MODEL_PATH}...")
    processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)
    model.to("cuda")
    model.eval()
    return model, processor

def run_ocr(model, processor, image_path):
    """Run OCR on a single image"""
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess
        pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda")
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=64)
        
        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    model, processor = load_model()
    
    # Test on sample images
    test_images = [
        '1264_199_crop_0.jpg',
        '1264_199_crop_1.jpg', 
        '1264_199_crop_2.jpg',
        '141_004_crop_1.jpg',
    ]
    
    print(f"\nTesting TrOCR Base (untrained) on {len(test_images)} images...")
    print("=" * 80)
    
    for img_name in test_images:
        img_path = os.path.join(CROPS_DIR, img_name)
        if os.path.exists(img_path):
            result = run_ocr(model, processor, img_path)
            print(f"\n{img_name}:")
            print(f"  Output: {result}")
        else:
            print(f"\n{img_name}: File not found")
    
    print("\n" + "=" * 80)
    print("Note: TrOCR Base is untrained on Han Nom, so outputs are random/English.")
    print("To get good results, you need to fine-tune it on your Han Nom dataset.")

if __name__ == "__main__":
    main()
