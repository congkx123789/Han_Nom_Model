import json
import os
import random
from tqdm import tqdm

def prepare_nomnaocr_data(patches_dir, labels_file, output_list):
    """Convert NomNaOCR patches to Qwen2.5-VL format."""
    print(f"Processing NomNaOCR labels from {labels_file}...")
    with open(labels_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in tqdm(lines):
        parts = line.strip().split('\t')
        if len(parts) == 2:
            img_rel_path, label = parts
            img_path = os.path.join(patches_dir, img_rel_path)
            if os.path.exists(img_path):
                entry = {
                    "id": f"nomnaocr_{len(output_list)}",
                    "image": img_path,
                    "conversations": [
                        {"from": "user", "value": "<image>\nOCR this Han Nom text."},
                        {"from": "assistant", "value": label}
                    ]
                }
                output_list.append(entry)

def prepare_character_data(char_dir, output_list):
    """Convert character dataset to Qwen2.5-VL format."""
    print(f"Processing character dataset from {char_dir}...")
    for char_hex in tqdm(os.listdir(char_dir)):
        char_path = os.path.join(char_dir, char_hex)
        if os.path.isdir(char_path):
            try:
                char = chr(int(char_hex, 16))
                for img_name in os.listdir(char_path):
                    img_path = os.path.join(char_path, img_name)
                    entry = {
                        "id": f"char_{len(output_list)}",
                        "image": img_path,
                        "conversations": [
                            {"from": "user", "value": "<image>\nIdentify this Han Nom character."},
                            {"from": "assistant", "value": char}
                        ]
                    }
                    output_list.append(entry)
            except ValueError:
                continue

def main():
    patches_dir = "data/NomNaOCR_dataset/Patches"
    train_labels = os.path.join(patches_dir, "Train.txt")
    val_labels = os.path.join(patches_dir, "Validate.txt")
    char_dir = "dataset"
    
    train_data = []
    val_data = []
    
    # Process NomNaOCR
    if os.path.exists(train_labels):
        prepare_nomnaocr_data(patches_dir, train_labels, train_data)
    if os.path.exists(val_labels):
        prepare_nomnaocr_data(patches_dir, val_labels, val_data)
        
    # Process Character Dataset (add to train)
    if os.path.exists(char_dir):
        prepare_character_data(char_dir, train_data)
        
    # Shuffle train data
    random.shuffle(train_data)
    
    # Save to JSONL
    os.makedirs("data/qwen_prepared", exist_ok=True)
    
    with open("data/qwen_prepared/train.jsonl", "w", encoding="utf-8") as f:
        for entry in train_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    with open("data/qwen_prepared/val.jsonl", "w", encoding="utf-8") as f:
        for entry in val_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            
    print(f"Saved {len(train_data)} training samples and {len(val_data)} validation samples.")

if __name__ == "__main__":
    main()
