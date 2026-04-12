from unsloth import FastVisionModel
import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_vl_utils import process_vision_info

class QwenDataset(Dataset):
    def __init__(self, jsonl_file, processor):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item["image"]
        label = item["conversations"][1]["value"]
        image = Image.open(image_path).convert("RGB")
        
        # Rotate vertical images to horizontal as requested by user
        if image.height > image.width:
            image = image.transpose(Image.Transpose.ROTATE_90)
            
        # Resize to a fixed size to ensure consistent patch count
        # Qwen2.5-VL uses patches, so different sizes lead to different tensor shapes
        image = image.resize((224, 224)) 
        
        # Qwen2.5-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR this Han Nom text."},
                ],
            }
        ]
        
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        
        # Prepare labels
        # We want to train on the assistant response only
        # For simplicity in this script, we'll just use the whole sequence and mask the user part
        # A more robust implementation would mask the prompt
        labels = self.processor.tokenizer(label, return_tensors="pt").input_ids
        
        # This is a simplified version. In a real scenario, we'd combine them properly.
        # For now, let's just return the processed inputs and add labels.
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # We need to append the label to the input_ids and create labels tensor
        # However, Trainer expects 'labels' to be the same shape as 'input_ids'
        full_text = text + label + self.processor.tokenizer.eos_token
        full_inputs = self.processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        )
        
        full_inputs = {k: v.squeeze(0) for k, v in full_inputs.items()}
        labels = full_inputs["input_ids"].clone()
        
        # Mask the prompt part in labels
        prompt_inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
        prompt_len = prompt_inputs.input_ids.shape[1]
        labels[:prompt_len] = -100
        
        # Mask padding tokens (where attention_mask is 0) to avoid calculating loss on padding
        if "attention_mask" in full_inputs:
            labels[full_inputs["attention_mask"] == 0] = -100
            
        full_inputs["labels"] = labels
        
        return full_inputs

import argparse

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL-3B-4bit on Han-Nom data with Unsloth")
    parser.add_argument("--model_path", type=str, default="./models/Qwen2.5-VL-3B-4bit", help="Path to the pre-trained model")
    parser.add_argument("--train_file", type=str, default="data/qwen_prepared/train.jsonl", help="Path to the training JSONL file")
    parser.add_argument("--val_file", type=str, default="data/qwen_prepared/val.jsonl", help="Path to the validation JSONL file")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/qwen2.5-vl-han-nom", help="Directory to save checkpoints")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of training steps (for testing)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Logging tool (tensorboard, none)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from, or 'True' to resume from latest")
    args = parser.parse_args()

    model_path = args.model_path
    train_file = args.train_file
    val_file = args.val_file
    output_dir = args.output_dir
    
    # Handle resume_from_checkpoint argument
    resume_from_checkpoint = args.resume_from_checkpoint
    if resume_from_checkpoint and resume_from_checkpoint.lower() == "true":
        resume_from_checkpoint = True
    
    # Load model and processor with Unsloth
    model, processor = FastVisionModel.from_pretrained(
        model_name = model_path,
        load_in_4bit = True,
        use_gradient_checkpointing = "unsloth", # Optimized gradient checkpointing
    )
    
    # Expand vocabulary with Han Nom characters
    print("Scanning dataset and dictionaries for unique characters to expand vocabulary...")
    unique_chars = set()
    
    # 1. Scan training data
    print("Scanning training data...")
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            label = data["conversations"][1]["value"]
            for char in label:
                unique_chars.add(char)
                
    # 2. Scan Thieu Chuu Dictionary
    thieu_chuu_path = "data/Thieu_Chuu_Dictionary.csv"
    if os.path.exists(thieu_chuu_path):
        print(f"Scanning {thieu_chuu_path}...")
        import csv
        with open(thieu_chuu_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "char" in row and row["char"]:
                    unique_chars.add(row["char"])
                    
    # 3. Scan Unihan Vietnamese
    unihan_path = "data/Unihan_Vietnamese.csv"
    if os.path.exists(unihan_path):
        print(f"Scanning {unihan_path}...")
        import csv
        with open(unihan_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "char" in row and row["char"]:
                    unique_chars.add(row["char"])

    print(f"Found {len(unique_chars)} unique characters in total.")
    
    new_tokens = []
    for char in unique_chars:
        if len(processor.tokenizer.encode(char, add_special_tokens=False)) > 1:
            new_tokens.append(char)
            
    if new_tokens:
        print(f"Adding {len(new_tokens)} new tokens to tokenizer...")
        num_added_toks = processor.tokenizer.add_tokens(new_tokens)
        print(f"Actually added {num_added_toks} tokens.")
        
        # Resize model embeddings
        model.resize_token_embeddings(len(processor.tokenizer))
        print(f"Resized model embeddings to {len(processor.tokenizer)}")
        
        # Smart Initialization for new tokens
        # Initialize new tokens with the mean of existing tokens to prevent loss shock
        print("Applying smart initialization for new tokens...")
        input_embeddings = model.get_input_embeddings()
        weights = input_embeddings.weight.data
        # Calculate mean of existing tokens (excluding the new ones)
        mean_embedding = torch.mean(weights[:-len(new_tokens), :], dim=0)
        # Assign mean to new tokens
        weights[-len(new_tokens):, :] = mean_embedding
        print("Smart initialization complete: New tokens initialized with mean embedding.")
    
    # Apply LoRA with Unsloth optimizations
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # True if you want to finetune the vision tower
        finetune_language_layers   = True,
        finetune_attention_modules = True,
        finetune_mlp_modules       = True,
        r = 16,
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        random_state = 3407,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj",],
        modules_to_save = ["embed_tokens", "lm_head"], # Needed for added tokens
    )
    
        # Let's add modules_to_save to LoraConfig.
        
    model.print_trainable_parameters()
    
    # Datasets
    train_dataset = QwenDataset(train_file, processor)
    val_dataset = QwenDataset(val_file, processor)
    
    # Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=50,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        logging_steps=1,
        eval_strategy="steps" if args.max_steps == -1 else "no",
        eval_steps=250,
        save_steps=250,
        save_total_limit=2,
        bf16=True,
        push_to_hub=False,
        report_to=args.report_to,
        logging_dir=os.path.join(output_dir, "logs"),
        remove_unused_columns=False,
        optim="adamw_8bit",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        group_by_length=True,
        lr_scheduler_type="cosine",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print(f"Starting training with Unsloth. Logging to {args.report_to}...")
    if resume_from_checkpoint:
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    # Save the adapter
    model.save_pretrained(os.path.join(output_dir, "final_adapter"))
    processor.save_pretrained(os.path.join(output_dir, "final_adapter"))
    print(f"Training complete. Adapter saved to {output_dir}/final_adapter")

if __name__ == "__main__":
    main()
