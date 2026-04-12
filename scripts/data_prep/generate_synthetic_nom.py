"""
Generate synthetic Chữ Nôm training images using NomNaTong font.
Creates 50,000+ training images for TrOCR fine-tuning.
"""

import os
import csv
import random
import json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

# Configuration
FONT_PATH = "fonts/NomNaTong-Regular.ttf"
FALLBACK_FONT = "NomNaTong-Regular.ttf"
DICT_FILE = "data/Thieu_Chuu_Dictionary.csv"
TRAIN_FILE = "data/got_ocr_dataset/train.txt"
VAL_FILE = "data/got_ocr_dataset/val.txt"
OUTPUT_DIR = "data/synthetic_nom"
NUM_IMAGES = 50000
VAL_RATIO = 0.1

# Image generation settings
IMG_HEIGHTS = [32, 48, 64]
FONT_SIZES = [24, 28, 32, 36, 40]
MAX_CHARS = 12
MIN_CHARS = 1

def load_font(size):
    """Load NomNaTong font"""
    for path in [FONT_PATH, FALLBACK_FONT]:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except:
                pass
    print(f"Warning: Could not load font, using default")
    return ImageFont.load_default()

def collect_all_chars():
    """Collect all Han Nom characters from dictionary and training data"""
    chars = set()
    
    # From dictionary
    if os.path.exists(DICT_FILE):
        with open(DICT_FILE, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                c = row.get('char', '').strip()
                if c and len(c) == 1:
                    chars.add(c)
    
    # From training data
    for f in [TRAIN_FILE, VAL_FILE]:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as fh:
                for line in fh:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        for c in parts[1]:
                            if ord(c) >= 0x3400:  # CJK range
                                chars.add(c)
    
    # From standard-nom
    nom_file = "data/chunom/standard-nom.csv"
    if os.path.exists(nom_file):
        with open(nom_file, 'r', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row[0].strip()) == 1:
                    chars.add(row[0].strip())
    
    return sorted(list(chars))

def can_render_char(font, char):
    """Check if font can render this character (not a tofu box)"""
    try:
        img = Image.new('L', (50, 50), 255)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), char, font=font, fill=0)
        # Check if anything was actually drawn
        pixels = list(img.getdata())
        return min(pixels) < 200  # Some dark pixels exist
    except:
        return False

def filter_renderable_chars(chars, font):
    """Filter to only chars the font can render"""
    renderable = []
    for c in chars:
        if can_render_char(font, c):
            renderable.append(c)
    return renderable

def generate_image(text, font, img_height=48):
    """Generate a single synthetic OCR image"""
    # Calculate text size
    temp_img = Image.new('RGB', (1000, 200), (255, 255, 255))
    temp_draw = ImageDraw.Draw(temp_img)
    bbox = temp_draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    
    # Create image with padding
    padding_x = random.randint(4, 12)
    padding_y = random.randint(2, 8)
    img_w = text_w + 2 * padding_x
    img_h = max(text_h + 2 * padding_y, img_height)
    
    # Random background (white to light gray)
    bg = random.randint(230, 255)
    img = Image.new('RGB', (img_w, img_h), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    
    # Random text color (black to dark gray)
    fg = random.randint(0, 50)
    
    # Center text
    x = padding_x - bbox[0]
    y = (img_h - text_h) // 2 - bbox[1]
    draw.text((x, y), text, font=font, fill=(fg, fg, fg))
    
    # Optional augmentations
    if random.random() < 0.3:
        # Add slight noise
        import numpy as np
        arr = np.array(img)
        noise = np.random.normal(0, random.randint(3, 10), arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
    
    if random.random() < 0.2:
        # Slight blur
        from PIL import ImageFilter
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))
    
    return img

def main():
    print("="*80)
    print("Synthetic Chữ Nôm Image Generator")
    print("="*80)
    
    # Collect characters
    print("\n[1/4] Collecting characters...")
    all_chars = collect_all_chars()
    print(f"  Total characters: {len(all_chars)}")
    
    # Load font and filter
    print("\n[2/4] Loading font and filtering renderable chars...")
    test_font = load_font(32)
    renderable = filter_renderable_chars(all_chars, test_font)
    print(f"  Renderable characters: {len(renderable)}")
    
    if len(renderable) < 100:
        print("ERROR: Too few renderable characters. Check font path.")
        return
    
    # Create output directories
    os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
    
    # Generate images
    print(f"\n[3/4] Generating {NUM_IMAGES} images...")
    
    train_data = []
    val_data = []
    
    # Character frequency weighting (common chars appear more)
    char_weights = [1.0] * len(renderable)
    
    for i in range(NUM_IMAGES):
        # Random text length
        text_len = random.choices(
            range(MIN_CHARS, MAX_CHARS + 1),
            weights=[10, 8, 6, 5, 4, 3, 2, 2, 1, 1, 1, 1],  # Bias towards shorter
            k=1
        )[0]
        
        # Random characters
        text = ''.join(random.choices(renderable, weights=char_weights, k=text_len))
        
        # Random font size and height
        font_size = random.choice(FONT_SIZES)
        img_height = random.choice(IMG_HEIGHTS)
        font = load_font(font_size)
        
        # Generate image
        img = generate_image(text, font, img_height)
        
        # Save
        img_name = f"syn_{i:06d}.jpg"
        img_path = os.path.join(OUTPUT_DIR, "images", img_name)
        img.save(img_path, quality=95)
        
        # Split train/val
        full_path = os.path.abspath(img_path)
        if random.random() < VAL_RATIO:
            val_data.append((full_path, text))
        else:
            train_data.append((full_path, text))
        
        if (i + 1) % 5000 == 0:
            print(f"  Generated {i+1}/{NUM_IMAGES} images...")
    
    # Write manifest files
    print(f"\n[4/4] Writing manifest files...")
    
    train_file = f"{OUTPUT_DIR}/train.txt"
    val_file = f"{OUTPUT_DIR}/val.txt"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for path, text in train_data:
            f.write(f"{path}\t{text}\n")
    
    with open(val_file, 'w', encoding='utf-8') as f:
        for path, text in val_data:
            f.write(f"{path}\t{text}\n")
    
    # Also create combined dataset (real + synthetic)
    combined_train = f"{OUTPUT_DIR}/combined_train.txt"
    combined_val = f"{OUTPUT_DIR}/combined_val.txt"
    
    with open(combined_train, 'w', encoding='utf-8') as f:
        # Real data first
        if os.path.exists(TRAIN_FILE):
            with open(TRAIN_FILE, 'r', encoding='utf-8') as rf:
                f.write(rf.read())
        # Then synthetic
        for path, text in train_data:
            f.write(f"{path}\t{text}\n")
    
    with open(combined_val, 'w', encoding='utf-8') as f:
        if os.path.exists(VAL_FILE):
            with open(VAL_FILE, 'r', encoding='utf-8') as rf:
                f.write(rf.read())
        for path, text in val_data:
            f.write(f"{path}\t{text}\n")
    
    print(f"\n{'='*80}")
    print(f"Generation complete!")
    print(f"{'='*80}")
    print(f"Synthetic train: {len(train_data)} images")
    print(f"Synthetic val: {len(val_data)} images")
    
    real_train = sum(1 for _ in open(TRAIN_FILE)) if os.path.exists(TRAIN_FILE) else 0
    real_val = sum(1 for _ in open(VAL_FILE)) if os.path.exists(VAL_FILE) else 0
    
    print(f"\nCombined dataset:")
    print(f"  Train: {real_train + len(train_data)} (real: {real_train}, synthetic: {len(train_data)})")
    print(f"  Val: {real_val + len(val_data)} (real: {real_val}, synthetic: {len(val_data)})")
    print(f"\nFiles:")
    print(f"  {combined_train}")
    print(f"  {combined_val}")

if __name__ == "__main__":
    main()
