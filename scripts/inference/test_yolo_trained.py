from ultralytics import YOLO
import os
import glob
from PIL import Image, ImageDraw
from tqdm import tqdm

# Paths
model_path = 'runs/detect/runs/detect/yolov8n_hannom/weights/best.pt' # Path to best model
image_dir = 'data/raw/images'
output_dir = 'data/yolo_inference_full'

os.makedirs(output_dir, exist_ok=True)

def draw_boxes(image_path, boxes, save_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    
    for box in boxes:
        # box: [x1, y1, x2, y2]
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        
    image.save(save_path)

def save_crops(image_path, boxes, base_output_dir):
    image = Image.open(image_path).convert("RGB")
    img_name = os.path.basename(image_path).split('.')[0]
    crop_dir = os.path.join(base_output_dir, f"{img_name}_crops")
    os.makedirs(crop_dir, exist_ok=True)
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        # Ensure coordinates are within bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.width, x2)
        y2 = min(image.height, y2)
        
        if x2 > x1 and y2 > y1:
            crop = image.crop((x1, y1, x2, y2))
            crop.save(os.path.join(crop_dir, f"crop_{i}.jpg"))

def main():
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Training might still be in progress.")
        return

    print(f"Loading model from {model_path}...")
    model = YOLO(model_path)
    
    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    
    print(f"Running inference on {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        img_name = os.path.basename(img_path)
        
        results = model(img_path, verbose=False)
        
        # Extract boxes
        boxes = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist() # [x1, y1, x2, y2]
                conf = box.conf[0].item()
                if conf > 0.25: # Threshold
                    boxes.append(coords)
        
        # Draw and save visualization
        save_path = os.path.join(output_dir, f"det_{img_name}")
        draw_boxes(img_path, boxes, save_path)
        
        # Save crops
        save_crops(img_path, boxes, output_dir)

    print(f"Processing complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
