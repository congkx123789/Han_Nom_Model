"""
Fine-tune TrOCR on NomNaOCR + Synthetic data.
Uses trocr-base-handwritten (better feature extraction than stage1).
End-to-end training, no freezing, proper weight surgery.
"""

import torch
import csv
import numpy as np
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset
from PIL import Image
import os
import evaluate

# Configuration
BASE_MODEL = "models/trocr_base_handwritten"
TRAIN_FILE = "data/synthetic_nom/combined_train.txt"
VAL_FILE = "data/synthetic_nom/combined_val.txt"
DICT_FILE = "data/Thieu_Chuu_Dictionary.csv"
OUTPUT_DIR = "models/trocr_hannom"
MAX_LENGTH = 64

def collect_han_nom_chars():
    all_chars = set()
    for f in [TRAIN_FILE, VAL_FILE, "data/got_ocr_dataset/train.txt", "data/got_ocr_dataset/val.txt"]:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        all_chars.update(parts[1])
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, 'r', encoding='utf-8-sig') as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                c = row.get('char', '').strip()
                if c: all_chars.add(c)
    nom_file = "data/chunom/standard-nom.csv"
    if os.path.exists(nom_file):
        with open(nom_file, 'r', encoding='utf-8-sig') as fh:
            for row in csv.reader(fh):
                if row: all_chars.add(row[0].strip())
    
    han_chars = set()
    for c in all_chars:
        cp = ord(c)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F or
            0x2B740 <= cp <= 0x2B81F or 0xF900 <= cp <= 0xFAFF or
            0x2F800 <= cp <= 0x2FA1F):
            han_chars.add(c)
    return han_chars

def expand_tokenizer_with_weight_surgery(processor, model):
    tokenizer = processor.tokenizer
    all_han_chars = collect_han_nom_chars()
    print(f"  Total Han Nom characters found: {len(all_han_chars)}")
    
    unk_id = tokenizer.unk_token_id
    new_tokens = []
    for c in sorted(all_han_chars):
        ids = tokenizer.encode(c, add_special_tokens=False)
        if unk_id in ids or len(ids) > 1:
            new_tokens.append(c)
    
    old_vocab_size = len(tokenizer)
    print(f"  Unknown to tokenizer: {len(new_tokens)}")
    print(f"  Original vocab size: {old_vocab_size}")
    
    if not new_tokens:
        return processor, model
    
    num_added = tokenizer.add_tokens(new_tokens)
    new_vocab_size = len(tokenizer)
    print(f"  Added {num_added} new tokens → vocab size: {new_vocab_size}")
    
    # Weight surgery: preserve old weights, init new ones properly
    old_input_embed = model.decoder.get_input_embeddings()
    old_input_weight = old_input_embed.weight.data.clone()
    old_output_embed = model.decoder.get_output_embeddings()
    old_output_weight = old_output_embed.weight.data.clone() if old_output_embed else None
    old_output_bias = old_output_embed.bias.data.clone() if (old_output_embed and old_output_embed.bias is not None) else None
    
    model.decoder.resize_token_embeddings(new_vocab_size)
    
    with torch.no_grad():
        new_input_embed = model.decoder.get_input_embeddings()
        new_input_embed.weight[:old_vocab_size] = old_input_weight
        std = old_input_weight.std().item()
        new_input_embed.weight[old_vocab_size:] = torch.randn(num_added, old_input_weight.shape[1]) * std * 0.02
        
        new_output_embed = model.decoder.get_output_embeddings()
        if new_output_embed is not None and old_output_weight is not None:
            new_output_embed.weight[:old_vocab_size] = old_output_weight
            new_output_embed.weight[old_vocab_size:] = torch.randn(num_added, old_output_weight.shape[1]) * std * 0.02
            if old_output_bias is not None and new_output_embed.bias is not None:
                new_output_embed.bias[:old_vocab_size] = old_output_bias
                new_output_embed.bias[old_vocab_size:] = 0.0
    
    print(f"  ✓ Weight surgery complete")
    
    model.config.decoder.vocab_size = new_vocab_size
    model.config.vocab_size = new_vocab_size
    if not hasattr(model.config, 'vocab_size'):
        setattr(model.config, 'vocab_size', new_vocab_size)
    
    return processor, model

class TrOCRDataset(Dataset):
    def __init__(self, data_file, processor, max_length=64):
        self.data = []
        self.processor = processor
        self.max_length = max_length
        
        print(f"Loading {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_path, text = parts
                    if os.path.exists(img_path):
                        self.data.append((img_path, text))
        print(f"  Loaded {len(self.data)} samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, text = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        if image.height > image.width * 1.5:
            image = image.transpose(Image.Transpose.ROTATE_90)
        
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text, padding="max_length", max_length=self.max_length,
            truncation=True, return_tensors="pt"
        ).input_ids
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        return {"pixel_values": pixel_values.squeeze(), "labels": labels.squeeze()}

def compute_metrics(pred):
    cer_metric = evaluate.load("cer")
    labels_ids = pred.label_ids
    pred_ids = pred.predictions
    vocab_size = len(processor.tokenizer)
    pred_ids = np.clip(pred_ids, 0, vocab_size - 1)
    try:
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)
        cer = cer_metric.compute(predictions=pred_str, references=label_str)
    except Exception as e:
        print(f"Warning: CER failed: {e}")
        cer = 1.0
    return {"cer": cer}

def main():
    global processor
    
    print("="*80)
    print("TrOCR Han Nom Fine-tuning")
    print("Base: trocr-base-handwritten | End-to-end | 47k images")
    print("="*80)
    
    # Load
    print("\n[1/4] Loading model...")
    processor = TrOCRProcessor.from_pretrained(BASE_MODEL)
    model = VisionEncoderDecoderModel.from_pretrained(BASE_MODEL)
    
    # Expand vocab
    print("\n[2/4] Expanding vocabulary (weight surgery)...")
    processor, model = expand_tokenizer_with_weight_surgery(processor, model)
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    
    # All params trainable (end-to-end, no freezing)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} (100%)")
    
    # Datasets
    print("\n[3/4] Loading datasets...")
    train_dataset = TrOCRDataset(TRAIN_FILE, processor, MAX_LENGTH)
    val_dataset = TrOCRDataset(VAL_FILE, processor, MAX_LENGTH)
    
    # Train end-to-end with single low LR
    print("\n[4/4] Starting training...")
    print("="*80)
    
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,        # Low LR for end-to-end
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=1000,
        eval_steps=1000,
        save_total_limit=3,
        bf16=True,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH,
        report_to="tensorboard",
        dataloader_num_workers=4,
    )
    
    trainer = Seq2SeqTrainer(
        model=model, args=training_args,
        train_dataset=train_dataset, eval_dataset=val_dataset,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # Save
    print("\nSaving final model...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*80)
    print(f"Training complete! Model saved to: {OUTPUT_DIR}")
    print("="*80)

if __name__ == "__main__":
    main()
