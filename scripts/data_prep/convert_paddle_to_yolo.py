import os
import json
import shutil
from PIL import Image
from tqdm import tqdm
import numpy as np

# Configuration
dataset_root = 'data/NomNaOCR_dataset/Pages'
output_root = 'data/yolo_dataset'
train_file = 'PaddleOCR-Train.txt'
val_file = 'PaddleOCR-Validate.txt'

# Create directories
for split in ['train', 'val']:
    os.makedirs(os.path.join(output_root, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, split, 'labels'), exist_ok=True)

def convert_to_yolo_bbox(points, img_width, img_height):
    # points: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
    # YOLO format: x_center, y_center, width, height (normalized)
    
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)
    
    # Clip to image bounds
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(img_width, xmax)
    ymax = min(img_height, ymax)
    
    bbox_w = xmax - xmin
    bbox_h = ymax - ymin
    
    x_center = xmin + bbox_w / 2
    y_center = ymin + bbox_h / 2
    
    # Normalize
    x_center /= img_width
    y_center /= img_height
    bbox_w /= img_width
    bbox_h /= img_height
    
    return x_center, y_center, bbox_w, bbox_h

def process_dataset(label_file, split):
    print(f"Processing {split} set from {label_file}...")
    
    with open(os.path.join(dataset_root, label_file), 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in tqdm(lines):
        try:
            # Parse line: path/to/img.jpg\tjson_string
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
                
            rel_img_path = parts[0]
            json_str = parts[1]
            
            # Absolute image path
            abs_img_path = os.path.join(dataset_root, rel_img_path)
            
            if not os.path.exists(abs_img_path):
                # Try decoding URL encoded characters if any (though usually not needed for local files)
                # Or check if path needs adjustment
                # print(f"Warning: Image not found: {abs_img_path}")
                continue
                
            # Read image to get dimensions
            try:
                with Image.open(abs_img_path) as img:
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error reading image {abs_img_path}: {e}")
                continue
                
            # Parse labels
            labels = json.loads(json_str)
            yolo_labels = []
            
            for label in labels:
                points = label['points']
                xc, yc, w, h = convert_to_yolo_bbox(points, img_width, img_height)
                
                # Class ID 0 for 'text'
                yolo_labels.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
            # Define output paths
            img_filename = os.path.basename(rel_img_path)
            # Handle potential duplicate filenames by including parent dir in name if needed
            # For now, assuming filenames are unique enough or we prepend parent dir
            # Let's prepend the parent directory name to be safe
            parent_dir = os.path.basename(os.path.dirname(rel_img_path))
            new_filename = f"{parent_dir}_{img_filename}"
            
            dst_img_path = os.path.join(output_root, split, 'images', new_filename)
            dst_label_path = os.path.join(output_root, split, 'labels', new_filename.replace('.jpg', '.txt').replace('.png', '.txt'))
            
            # Copy image
            shutil.copy2(abs_img_path, dst_img_path)
            
            # Write label file
            with open(dst_label_path, 'w', encoding='utf-8') as out_f:
                out_f.write('\n'.join(yolo_labels))
                
        except Exception as e:
            print(f"Error processing line: {line[:50]}... : {e}")

# Process Train and Val
process_dataset(train_file, 'train')
process_dataset(val_file, 'val')

# Create data.yaml
yaml_content = f"""
path: {os.path.abspath(output_root)} # dataset root dir
train: train/images # train images (relative to 'path')
val: val/images # val images (relative to 'path')

# Classes
names:
  0: text
"""

with open(os.path.join(output_root, 'data.yaml'), 'w', encoding='utf-8') as f:
    f.write(yaml_content)

print(f"Dataset preparation complete at {output_root}")
