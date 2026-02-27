"""
Test different OCR prompts on the same images.
Compare: Role-play, Few-shot, Context-based prompts.
"""

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import os
from PIL import Image
import gc

# Configuration
QWEN_BASE_MODEL = 'models/Qwen2.5-VL-3B'

# Test images with ground truth
TEST_IMAGES = [
    ('1264_199_crop_0.jpg', '阿修羅中碼瑙色'),
    ('1264_199_crop_1.jpg', '兜率天上閣浮金'),
    ('1264_199_crop_2.jpg', '化樂天中大鼓音'),
    ('1035_002_crop_2.jpg', '面不...建...'),  # Partial ground truth
]

# Different prompts to test
PROMPTS = {
    'original': "Hãy đọc văn bản Hán Nôm trong ảnh này.",
    
    'roleplay': """Bạn là một công cụ OCR chuyên dụng cho văn bản Hán Nôm cổ. Nhiệm vụ duy nhất của bạn là trích xuất các ký tự có trong ảnh.

Yêu cầu bắt buộc:
- Đọc văn bản theo chiều dọc (từ trên xuống dưới, từ phải sang trái).
- Chỉ trả về chuỗi ký tự text.
- KHÔNG thêm bất kỳ lời dẫn, giải thích, hay nhận xét nào.
- Nếu chữ quá mờ không đọc được, hãy dùng ký tự '□'.

Hãy bắt đầu ngay:""",

    'fewshot': """Hãy chuyển đổi hình ảnh văn bản Hán Nôm sau thành text.

Ví dụ format mong muốn:
Input: [Ảnh văn bản Hán Nôm]
Output: 阿修羅中碼瑙色

Bây giờ hãy làm tương tự với ảnh này. Chỉ xuất ra text kết quả:""",

    'context': """Trích xuất văn bản từ hình ảnh kinh sách/y thư cổ này. 
Chú ý các thuật ngữ Phật giáo/Đông y. 
Văn bản viết dọc từ trên xuống, từ phải sang trái.
Chỉ trả về nội dung chữ Hán, không giải thích gì thêm.""",

    'minimal': """OCR. Chỉ output ký tự Hán. Không giải thích.""",
}

def load_model():
    print(f"Loading Qwen base model from {QWEN_BASE_MODEL}...")
    processor = AutoProcessor.from_pretrained(QWEN_BASE_MODEL)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    return model, processor

def run_ocr(model, processor, image_path, prompt_text):
    image = Image.open(image_path).convert("RGB")
    
    # Rotate if vertical
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_90)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
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
            max_new_tokens=64,  # Shorter to force concise output
            repetition_penalty=1.2,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    del inputs, generated_ids
    torch.cuda.empty_cache()
    
    return output_text[0].strip() if output_text else ""

def main():
    crops_dir = 'data/yolo_qwen_ocr_results'
    
    model, processor = load_model()
    
    print("\n" + "="*100)
    print("PROMPT COMPARISON TEST")
    print("="*100)
    
    results = []
    
    for img_name, ground_truth in TEST_IMAGES:
        img_path = os.path.join(crops_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Image: {img_name}")
        print(f"Ground Truth: {ground_truth}")
        print(f"{'='*80}")
        
        for prompt_name, prompt_text in PROMPTS.items():
            output = run_ocr(model, processor, img_path, prompt_text)
            # Truncate for display
            output_display = output[:60] + "..." if len(output) > 60 else output
            print(f"  [{prompt_name:12}] {output_display}")
            
            results.append({
                'image': img_name,
                'ground_truth': ground_truth,
                'prompt': prompt_name,
                'output': output[:100],
            })
    
    # Save results
    import csv
    output_file = 'data/prompt_comparison.csv'
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'ground_truth', 'prompt', 'output'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()
