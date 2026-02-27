import os
import cv2
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import numpy as np

# Initialize PaddleOCR with default model (PP-OCRv5_server_det)
# Adjust parameters to improve coverage
ocr = PaddleOCR(
    use_textline_orientation=True, # Enable orientation classification
    lang='ch',
    det_db_unclip_ratio=2.0, # Expand boxes slightly
    det_db_box_thresh=0.5    # Threshold for box confidence
)

# Input and output directories
image_dir = 'data/raw/images'
output_dir = 'data/paddle_det_results'
os.makedirs(output_dir, exist_ok=True)

# List of sample images
image_files = [
    "1321_001.jpg", "1321_002.jpg", "1321_003.jpg", "1321_004.jpg", "1321_005.jpg",
    "1321_006.jpg", "1321_007.jpg", "1321_008.jpg", "1321_009.jpg", "1321_010.jpg"
]

def draw_boxes(image, boxes):
    draw = ImageDraw.Draw(image)
    for box in boxes:
        # box is a list of 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Convert to tuple of tuples for polygon, ensuring integers
        polygon = [tuple(map(int, point)) for point in box]
        # Set width to 1 (thinnest possible) as requested
        draw.polygon(polygon, outline='red', width=1)
    return image

print(f"Running detection on {len(image_files)} images...")

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
        
    print(f"Processing {img_file}...")
    
    # Run detection
    # Note: ocr() returns a list of OCRResult objects (one per image)
    results = ocr.ocr(img_path)
    
    if not results:
        print(f"No text detected in {img_file}")
        continue
        
    # Access the first result (since we processed one image)
    res = results[0]
    
    # Check if we have detection polygons
    # The object behaves like a dict
    if 'dt_polys' in res:
        boxes = res['dt_polys']
    elif 'boxes' in res:
        boxes = res['boxes']
    else:
        print(f"Could not find boxes in result keys: {res.keys()}")
        continue
        
    if boxes is None or len(boxes) == 0:
        print(f"No boxes found in {img_file}")
        continue

    # Draw results
    image = Image.open(img_path).convert('RGB')
    im_show = draw_boxes(image, boxes)
    
    # Save result with suffix to avoid caching issues
    save_path = os.path.join(output_dir, f"det_{img_file.replace('.jpg', '_tuned.jpg')}")
    im_show.save(save_path)
    print(f"Saved result to {save_path}")

    # Save cropped images for the first image only (to avoid clutter)
    if img_file == "1321_001.jpg":
        crop_dir = os.path.join(output_dir, "crops_1321_001")
        os.makedirs(crop_dir, exist_ok=True)
        for i, box in enumerate(boxes):
            # box is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Get min/max x and y to crop (axis-aligned crop for simplicity)
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            
            # Add some padding? unclip_ratio already expanded it, but let's be safe
            # Ensure within bounds
            left = max(0, int(left))
            top = max(0, int(top))
            right = min(image.width, int(right))
            bottom = min(image.height, int(bottom))
            
            crop = image.crop((left, top, right, bottom))
            crop.save(os.path.join(crop_dir, f"crop_{i}.jpg"))
        print(f"Saved {len(boxes)} cropped images to {crop_dir}")

print("Done.")

