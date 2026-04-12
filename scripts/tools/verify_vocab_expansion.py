from transformers import AutoProcessor
import json
import os
import csv

model_path = "./models/Qwen2.5-VL-3B-4bit"
train_file = "data/qwen_prepared/train.jsonl"
thieu_chuu_path = "data/Thieu_Chuu_Dictionary.csv"
unihan_path = "data/Unihan_Vietnamese.csv"

print(f"Loading processor from {model_path}...")
try:
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer
except Exception as e:
    print(f"Error loading processor: {e}")
    exit(1)

print("Scanning dataset and dictionaries for unique characters to expand vocabulary...")
unique_chars = set()

# 1. Scan training data
if os.path.exists(train_file):
    print(f"Scanning {train_file}...")
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            label = data["conversations"][1]["value"]
            for char in label:
                unique_chars.add(char)
else:
    print(f"Warning: {train_file} not found.")

# 2. Scan Thieu Chuu Dictionary
if os.path.exists(thieu_chuu_path):
    print(f"Scanning {thieu_chuu_path}...")
    with open(thieu_chuu_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "char" in row and row["char"]:
                unique_chars.add(row["char"])
else:
    print(f"Warning: {thieu_chuu_path} not found.")

# 3. Scan Unihan Vietnamese
if os.path.exists(unihan_path):
    print(f"Scanning {unihan_path}...")
    with open(unihan_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "char" in row and row["char"]:
                unique_chars.add(row["char"])
else:
    print(f"Warning: {unihan_path} not found.")

print(f"Found {len(unique_chars)} unique characters in total.")

print("Checking against current tokenizer...")
new_tokens = []
for char in unique_chars:
    if len(tokenizer.encode(char, add_special_tokens=False)) > 1:
        new_tokens.append(char)

print(f"Characters requiring new tokens: {len(new_tokens)}")
if len(new_tokens) > 0:
    print(f"Example new tokens: {new_tokens[:10]}")
    print(f"Adding {len(new_tokens)} new tokens to tokenizer...")
    num_added_toks = tokenizer.add_tokens(new_tokens)
    print(f"Actually added {num_added_toks} tokens.")
    print(f"New tokenizer size: {len(tokenizer)}")
else:
    print("No new tokens needed.")
