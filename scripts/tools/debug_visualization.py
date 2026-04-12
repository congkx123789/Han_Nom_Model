import os
import cv2
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw
import numpy as np

# Initialize PaddleOCR
ocr = PaddleOCR(
    use_textline_orientation=False,
    lang='ch'
)

img_file = "1321_001.jpg"
image_dir = 'data/raw/images'
output_dir = 'data/paddle_det_results'
img_path = os.path.join(image_dir, img_file)

print(f"Processing {img_file}...")

# Run detection
results = ocr.ocr(img_path)
res = results[0]

# Get boxes
if 'dt_polys' in res:
    boxes = res['dt_polys']
elif 'boxes' in res:
    boxes = res['boxes']
else:
    boxes = []

print(f"Found {len(boxes)} boxes")
if len(boxes) > 0:
    print(f"First box coordinates: {boxes[0]}")

# Draw
image = Image.open(img_path).convert('RGB')
print(f"Image size: {image.size}")

draw = ImageDraw.Draw(image)

for i, box in enumerate(boxes):
    # Convert to list of tuples
    polygon = [tuple(map(int, point)) for point in box]
    
    # Draw filled polygon with transparency? PIL doesn't support transparency on RGB directly easily without RGBA
    # Let's just draw thick outline in GREEN and a circle at the first point
    draw.polygon(polygon, outline='green', width=15)
    
    # Draw a circle at the first point to mark start
    p0 = polygon[0]
    r = 10
    draw.ellipse((p0[0]-r, p0[1]-r, p0[0]+r, p0[1]+r), fill='blue')
    
    print(f"Box {i}: {polygon}")

# Save
save_path = os.path.join(output_dir, f"debug_{img_file}")
image.save(save_path)
print(f"Saved debug result to {save_path}")
