from transformers import AutoProcessor
import json
from tqdm import tqdm

model_path = "./models/Qwen2.5-VL-3B-4bit"
train_file = "data/qwen_prepared/train.jsonl"

print(f"Loading processor from {model_path}...")
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = processor.tokenizer

print(f"Scanning {train_file} for unique characters...")
unique_chars = set()
with open(train_file, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        # The label is in conversations[1]['value']
        label = data["conversations"][1]["value"]
        for char in label:
            unique_chars.add(char)

print(f"Total unique characters in dataset: {len(unique_chars)}")

missing_chars = []
for char in tqdm(unique_chars):
    # Check if character is tokenized as a single token
    tokens = tokenizer.tokenize(char)
    ids = tokenizer.encode(char, add_special_tokens=False)
    
    # Ideally, a common character should be 1 token. 
    # If it splits into multiple bytes/tokens (and isn't just a byte fallback), it might be "unknown" or inefficient.
    # However, Qwen uses byte-level BPE. 
    # A better check: is the character explicitly in the vocab?
    # Or does it decompose into generic byte tokens?
    
    if len(ids) > 1:
        # It takes multiple tokens to represent this character
        missing_chars.append(char)

print(f"Characters requiring multiple tokens: {len(missing_chars)}")
print(f"Percentage of 'inefficient' characters: {len(missing_chars)/len(unique_chars)*100:.2f}%")

if len(missing_chars) > 0:
    print("Top 20 missing characters:", missing_chars[:20])
    
    # Save missing chars to file
    with open("missing_chars.txt", "w", encoding="utf-8") as f:
        for char in missing_chars:
            f.write(char + "\n")
    print("Saved missing characters to missing_chars.txt")
else:
    print("Tokenizer covers all characters efficiently!")
