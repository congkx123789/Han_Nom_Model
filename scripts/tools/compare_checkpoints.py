"""
Compare different Qwen checkpoints on the same cropped images.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import os
import glob
from PIL import Image
from tqdm import tqdm
import gc

# Configuration
QWEN_BASE_MODEL = 'models/Qwen2.5-VL-3B'
CROPS_DIR = 'data/yolo_qwen_ocr_results'
OUTPUT_FILE = 'data/checkpoint_comparison.csv'

# Checkpoints to compare
CHECKPOINTS = [
    ('checkpoint-1000', 'checkpoints/qwen2.5-vl-han-nom/checkpoint-1000'),
    ('checkpoint-750', 'checkpoints/qwen2.5-vl-han-nom/checkpoint-750'),
]

def load_qwen_model(base_model_path, checkpoint_path):
    """Load Qwen model with fine-tuned adapter"""
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Resize embeddings to match checkpoint
    model.resize_token_embeddings(171663, mean_resizing=False)
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    return model, processor

def run_qwen_ocr(model, processor, image_path):
    """Run OCR on a single image"""
    image = Image.open(image_path).convert("RGB")
    
    # Rotate if vertical text
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_90)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Hãy đọc văn bản Hán Nôm trong ảnh này."},
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            repetition_penalty=1.5,  # Increased to reduce hallucination
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Cleanup
    del inputs, generated_ids
    torch.cuda.empty_cache()
    
    return output_text[0] if output_text else ""

def main():
    # Get crop images
    crop_files = sorted(glob.glob(os.path.join(CROPS_DIR, '*.jpg')))[:10]  # Test on 10 crops
    print(f"Testing on {len(crop_files)} crop images")
    
    results = []
    
    for ckpt_name, ckpt_path in CHECKPOINTS:
        print(f"\n{'='*50}")
        print(f"Loading checkpoint: {ckpt_name}")
        print(f"{'='*50}")
        
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            continue
        
        model, processor = load_qwen_model(QWEN_BASE_MODEL, ckpt_path)
        
        for crop_path in tqdm(crop_files, desc=f"Processing with {ckpt_name}"):
            crop_name = os.path.basename(crop_path)
            ocr_text = run_qwen_ocr(model, processor, crop_path)
            results.append({
                'checkpoint': ckpt_name,
                'crop_file': crop_name,
                'ocr_text': ocr_text.strip().replace('\n', ' ')[:100],  # Limit length
            })
            print(f"  {crop_name}: {ocr_text[:50]}...")
        
        # Free memory
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    
    # Save results
    import csv
    with open(OUTPUT_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['checkpoint', 'crop_file', 'ocr_text'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to: {OUTPUT_FILE}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    
    # Group by crop file
    from collections import defaultdict
    by_crop = defaultdict(dict)
    for r in results:
        by_crop[r['crop_file']][r['checkpoint']] = r['ocr_text']
    
    for crop_name, ckpt_results in by_crop.items():
        print(f"\n{crop_name}:")
        for ckpt_name, text in ckpt_results.items():
            print(f"  {ckpt_name}: {text[:60]}...")

if __name__ == "__main__":
    main()
