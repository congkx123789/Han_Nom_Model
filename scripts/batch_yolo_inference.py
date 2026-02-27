"""
Batch YOLO inference on the full dataset from labels.csv
Saves detection results (visualizations and crops) to a separate folder.
"""

from ultralytics import YOLO
import os
import csv
from PIL import Image, ImageDraw
from tqdm import tqdm
import argparse

# Configuration
MODEL_PATH = 'runs/detect/runs/detect/yolov8n_hannom/weights/best.pt'
LABELS_CSV = 'data/raw/labels.csv'
OUTPUT_DIR = 'data/yolo_full_inference'

def draw_boxes(image, boxes):
    """Draw bounding boxes on image (in-memory)"""
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
    return image

def save_crops(image, boxes, crop_dir, prefix):
    """Save cropped text regions"""
    os.makedirs(crop_dir, exist_ok=True)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        
        if x2 > x1 and y2 > y1:
            crop = image.crop((x1, y1, x2, y2))
            crop.save(os.path.join(crop_dir, f"{prefix}_crop_{i}.jpg"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-crops', action='store_true', help='Save cropped text regions')
    parser.add_argument('--skip-existing', action='store_true', help='Skip already processed images')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of images to process (0 = all)')
    args = parser.parse_args()

    # Create output directories
    vis_dir = os.path.join(OUTPUT_DIR, 'visualizations')
    crops_dir = os.path.join(OUTPUT_DIR, 'crops')
    os.makedirs(vis_dir, exist_ok=True)
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)

    # Read image paths from labels.csv
    print(f"Reading image paths from {LABELS_CSV}...")
    image_paths = []
    with open(LABELS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            local_path = row.get('local_path', '')
            if local_path:
                image_paths.append(local_path)

    print(f"Found {len(image_paths)} images in labels.csv")
    
    if args.limit > 0:
        image_paths = image_paths[:args.limit]
        print(f"Limited to {len(image_paths)} images")

    # Process images
    processed = 0
    skipped = 0
    errors = 0

    for img_path in tqdm(image_paths, desc="Processing"):
        try:
            # Check if file exists
            if not os.path.exists(img_path):
                errors += 1
                continue

            # Generate output filename
            # Use volume_page format from path
            img_name = os.path.basename(img_path).replace('.jpg', '')
            vis_output = os.path.join(vis_dir, f"det_{img_name}.jpg")

            # Skip if already processed
            if args.skip_existing and os.path.exists(vis_output):
                skipped += 1
                continue

            # Run inference
            results = model(img_path, verbose=False)

            # Extract boxes
            boxes = []
            for result in results:
                for box in result.boxes:
                    coords = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    if conf > 0.25:
                        boxes.append(coords)

            # Load image for visualization
            image = Image.open(img_path).convert("RGB")

            # Draw boxes and save visualization
            vis_image = draw_boxes(image.copy(), boxes)
            vis_image.save(vis_output)

            # Save crops if requested
            if args.save_crops and boxes:
                crop_subdir = os.path.join(crops_dir, img_name)
                save_crops(image, boxes, crop_subdir, img_name)

            processed += 1

        except Exception as e:
            errors += 1
            if errors < 10:  # Only print first 10 errors
                print(f"\nError processing {img_path}: {e}")

    print(f"\n{'='*50}")
    print(f"Processing complete!")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Errors: {errors}")
    print(f"Results saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
