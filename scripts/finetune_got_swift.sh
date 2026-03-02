#!/bin/bash

# Fine-tune GOT-OCR 2.0 with ms-swift
# Optimized for RTX 5060 Ti (8GB VRAM)

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model_type got-ocr2 \
    --model_id_or_path stepfun-ai/GOT-OCR2_0 \
    --sft_type lora \
    --dataset data/got_swift_dataset/train.jsonl \
    --val_dataset data/got_swift_dataset/val.jsonl \
    --output_dir output/got_ocr_hannom_swift \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules q_proj v_proj \
    --gradient_accumulation_steps 16 \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 2 \
    --logging_steps 10 \
    --dataloader_num_workers 4 \
    --model_author "Antigravity" \
    --model_name "GOT-OCR2-HanNom"
