"""
Pipeline: YOLO Detection -> Crop -> Qwen OCR
Runs on random samples from the full dataset.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from ultralytics import YOLO
import os
import random
import csv
from PIL import Image
import argparse
from tqdm import tqdm

# Configuration
YOLO_MODEL_PATH = 'runs/detect/runs/detect/yolov8n_hannom/weights/best.pt'
QWEN_BASE_MODEL = 'models/Qwen2.5-VL-3B'
QWEN_CHECKPOINT = 'checkpoints/qwen2.5-vl-han-nom/checkpoint-1000'
LABELS_CSV = 'data/raw/labels.csv'
OUTPUT_DIR = 'data/yolo_qwen_ocr_results'

def load_qwen_model(base_model_path, checkpoint_path):
    """Load Qwen model with fine-tuned adapter"""
    print(f"Loading Qwen processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(base_model_path)
    
    print(f"Loading Qwen base model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Resize embeddings to match checkpoint
    print(f"Original embedding size: {model.get_input_embeddings().weight.shape}")
    model.resize_token_embeddings(171663, mean_resizing=False)
    print(f"Resized embedding size: {model.get_input_embeddings().weight.shape}")
    
    # Load LoRA adapter
    print(f"Loading adapter from {checkpoint_path}...")
    model = PeftModel.from_pretrained(model, checkpoint_path)
    model.eval()
    
    return model, processor

def run_qwen_ocr(model, processor, image):
    """Run OCR on a single image"""
    # Rotate if vertical text
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_90)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": """Bạn là một công cụ OCR chuyên dụng cho văn bản Hán Nôm cổ. Nhiệm vụ duy nhất của bạn là trích xuất các ký tự có trong ảnh.

Yêu cầu bắt buộc:
- Đọc văn bản theo chiều dọc (từ trên xuống dưới, từ phải sang trái).
- Chỉ trả về chuỗi ký tự text.
- KHÔNG thêm bất kỳ lời dẫn, giải thích, hay nhận xét nào.
- Nếu chữ quá mờ không đọc được, hãy dùng ký tự '□'.

Hãy bắt đầu ngay:"""},
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
            max_new_tokens=256,
            repetition_penalty=1.2,
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=10, help='Number of random images to process')
    parser.add_argument('--max-crops-per-image', type=int, default=5, help='Max crops per image')
    parser.add_argument('--conf-threshold', type=float, default=0.5, help='YOLO confidence threshold')
    args = parser.parse_args()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load YOLO model
    print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    
    # Load Qwen model
    qwen_model, processor = load_qwen_model(QWEN_BASE_MODEL, QWEN_CHECKPOINT)
    
    # Read image paths
    print(f"Reading image paths from {LABELS_CSV}...")
    image_paths = []
    with open(LABELS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = row.get('local_path', '')
            if local_path and os.path.exists(local_path):
                image_paths.append(local_path)
    
    print(f"Found {len(image_paths)} valid images")
    
    # Random sample
    samples = random.sample(image_paths, min(args.num_samples, len(image_paths)))
    print(f"Processing {len(samples)} random samples...")
    
    results = []
    
    for img_path in tqdm(samples, desc="Processing"):
        try:
            # Run YOLO detection
            yolo_results = yolo_model(img_path, verbose=False)
            
            # Extract boxes
            boxes = []
            for result in yolo_results:
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    if conf > args.conf_threshold:
                        boxes.append((coords, conf))
            
            if not boxes:
                print(f"No detections in {img_path}")
                continue
            
            # Sort by confidence, take top N
            boxes.sort(key=lambda x: x[1], reverse=True)
            boxes = boxes[:args.max_crops_per_image]
            
            # Load image
            image = Image.open(img_path).convert("RGB")
            img_name = os.path.basename(img_path).replace('.jpg', '')
            
            # Process each crop
            for i, (coords, conf) in enumerate(boxes):
                x1, y1, x2, y2 = map(int, coords)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.width, x2)
                y2 = min(image.height, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                
                crop = image.crop((x1, y1, x2, y2))
                
                # Run Qwen OCR
                ocr_text = run_qwen_ocr(qwen_model, processor, crop)
                
                # Save crop
                crop_filename = f"{img_name}_crop_{i}.jpg"
                crop_path = os.path.join(OUTPUT_DIR, crop_filename)
                crop.save(crop_path)
                
                results.append({
                    'source_image': img_path,
                    'crop_file': crop_filename,
                    'confidence': f"{conf:.3f}",
                    'ocr_text': ocr_text.strip().replace('\n', ' '),
                })
                
                print(f"  [{i}] conf={conf:.2f} -> {ocr_text[:50]}...")
        
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    
    # Save results to CSV
    results_csv = os.path.join(OUTPUT_DIR, 'ocr_results.csv')
    with open(results_csv, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['source_image', 'crop_file', 'confidence', 'ocr_text'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed {len(results)} text regions from {len(samples)} images")
    print(f"Results saved to: {results_csv}")
    print(f"Crops saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
