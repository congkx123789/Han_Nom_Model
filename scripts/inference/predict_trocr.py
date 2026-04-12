"""
Predict text from image using trained TrOCR model
"""

import argparse
import os
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from peft import PeftModel
from PIL import Image

def load_trained_model(adapter_path):
    print(f"Loading base processor from microsoft/trocr-base-stage1...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-stage1")
    
    print("Loading base model...")
    base_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
    
    print(f"Loading LoRA weights from {adapter_path}...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    model.to("cuda")
    model.eval()
    
    return model, processor

def predict_image(model, processor, image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        
        # Preprocess
        pixel_values = processor(image, return_tensors="pt").pixel_values.to("cuda")
        
        # Generate with generation arguments
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=64,
                num_beams=4,
                early_stopping=True,
                repetition_penalty=1.2
            )
        
        # Decode
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text.strip()
    except Exception as e:
        return f"Error predicting {image_path}: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Predict text using trained TrOCR model")
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--adapter", type=str, default="models/trocr_hannom_lora", help="Path to trained LoRA adapter")
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image {args.image} not found")
        return
        
    if not os.path.exists(args.adapter):
        print(f"Error: Adapter path {args.adapter} not found. Please train the model first.")
        return
        
    model, processor = load_trained_model(args.adapter)
    
    print(f"\nPredicting: {args.image}")
    result = predict_image(model, processor, args.image)
    print(f"\nResult:\n{result}")

if __name__ == "__main__":
    main()
