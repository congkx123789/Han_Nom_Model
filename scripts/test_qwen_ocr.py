import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import os
import argparse
from PIL import Image

def run_inference(args):
    # Load model and processor
    if args.checkpoint_path:
        # Load processor from base model path (processor files are not stored in checkpoint)
        processor = AutoProcessor.from_pretrained(args.model_path)
        # Load the base model (without LoRA) from the original model_path
        # IMPORTANT: Do NOT load from checkpoint_path here, as that triggers auto-adapter loading which fails due to size mismatch
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        # Resize model embeddings to match the checkpoint's vocab size (171663)
        print(f"Original embedding size: {model.get_input_embeddings().weight.shape}")
        # Use mean_resizing=False to avoid OOM during initialization of new embeddings
        model.resize_token_embeddings(171663, mean_resizing=False)
        print(f"Resized embedding size: {model.get_input_embeddings().weight.shape}")
        
        # Load the LoRA adapter
        model = PeftModel.from_pretrained(model, args.checkpoint_path)

        # Adapter is already merged in the checkpoint, no extra loading needed
    else:
        # Original behavior: load base model and optional LoRA adapter
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if args.adapter_path:
            print(f"Loading adapter from {args.adapter_path}...")
            model = PeftModel.from_pretrained(model, args.adapter_path)
    
    model.eval()
    
    # Prepare messages
    # Load image and rotate if necessary (matching training logic)
    image = Image.open(args.image_path).convert("RGB")
    if image.height > image.width:
        image = image.transpose(Image.Transpose.ROTATE_90)

    # Prepare messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "OCR this Han Nom text."},
            ],
        }
    ]
    
    # Process inputs
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    
    # Generate
    print("Generating OCR result...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.2)
                
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    # Move inputs to CPU to free GPU memory before decoding (if possible/needed)
    # Actually, we need them on GPU for generation.
    # After generation, we can delete them.
    del inputs
    del image_inputs
    del video_inputs
    torch.cuda.empty_cache()

    return output_text[0]
    
    del model
    del processor
    torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Test Qwen2.5-VL OCR on Han-Nom images")
    parser.add_argument("--model_path", type=str, default="./models/Qwen2.5-VL-3B-4bit", help="Path to base model (unused when --checkpoint_path is set)")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter (unused when --checkpoint_path is set)")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to full checkpoint directory (model + tokenizer + adapter)")
    parser.add_argument("--image_path", type=str, required=True, help="Path to image file")
    args = parser.parse_args()
    
    result = run_inference(args)
    print("\n" + "=" * 30)
    print("OCR Result:")
    print("-" * 30)
    print(result)
    print("=" * 30)

if __name__ == "__main__":
    main()
