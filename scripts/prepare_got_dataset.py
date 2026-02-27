"""
Prepare NomNaOCR dataset for GOT-OCR fine-tuning.
Converts PaddleOCR format to GOT-OCR format (image_path + text).
"""

import json
import os
from pathlib import Path
from tqdm import tqdm

# Paths
PADDLE_DIR = "data/NomNaOCR_dataset/Pages"
OUTPUT_DIR = "data/got_ocr_dataset"
TRAIN_FILE = os.path.join(PADDLE_DIR, "PaddleOCR-Train.txt")
VAL_FILE = os.path.join(PADDLE_DIR, "PaddleOCR-Validate.txt")

def parse_paddle_line(line):
    """Parse PaddleOCR format line"""
    parts = line.strip().split('\t')
    if len(parts) != 2:
        return None, None
    
    img_path = parts[0]
    annotations = json.loads(parts[1])
    
    # Extract all text from annotations
    texts = []
    for ann in annotations:
        text = ann.get('transcription', '').strip()
        if text and text != '###':  # Skip invalid annotations
            texts.append(text)
    
    # Combine all text (vertical text, top to bottom, right to left)
    full_text = ''.join(texts)
    
    return img_path, full_text

def convert_dataset(input_file, output_file):
    """Convert PaddleOCR format to GOT-OCR format"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    valid_count = 0
    skipped_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, desc=f"Converting {os.path.basename(input_file)}"):
            img_path, text = parse_paddle_line(line)
            
            if img_path and text:
                # Convert relative path to absolute
                full_img_path = os.path.join(PADDLE_DIR, img_path)
                
                # Check if image exists
                if os.path.exists(full_img_path):
                    # Write in GOT-OCR format: image_path\ttext
                    f_out.write(f"{full_img_path}\t{text}\n")
                    valid_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1
    
    return valid_count, skipped_count

def main():
    print("="*80)
    print("Preparing NomNaOCR dataset for GOT-OCR fine-tuning")
    print("="*80)
    
    # Convert training set
    train_output = os.path.join(OUTPUT_DIR, "train.txt")
    print(f"\n[1/2] Converting training set...")
    train_valid, train_skip = convert_dataset(TRAIN_FILE, train_output)
    print(f"  ✓ Valid: {train_valid}")
    print(f"  ✗ Skipped: {train_skip}")
    
    # Convert validation set
    val_output = os.path.join(OUTPUT_DIR, "val.txt")
    print(f"\n[2/2] Converting validation set...")
    val_valid, val_skip = convert_dataset(VAL_FILE, val_output)
    print(f"  ✓ Valid: {val_valid}")
    print(f"  ✗ Skipped: {val_skip}")
    
    print(f"\n{'='*80}")
    print(f"Dataset preparation complete!")
    print(f"{'='*80}")
    print(f"Training samples: {train_valid}")
    print(f"Validation samples: {val_valid}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"\nFiles created:")
    print(f"  - {train_output}")
    print(f"  - {val_output}")

if __name__ == "__main__":
    main()
