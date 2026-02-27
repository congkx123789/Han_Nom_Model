from PIL import Image
import os

image_path = "data/NomNaOCR_dataset/Patches/DVSKTT-5 Ban ky tuc bien/DVSKTT_ban_tuc_XVIII_45b_2.jpg"
output_path = "rotated_debug.jpg"

if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit(1)

print(f"Loading image: {image_path}")
image = Image.open(image_path).convert("RGB")
print(f"Original Size: {image.size} (Width x Height)")

if image.height > image.width:
    print("Image is vertical. Rotating 90 degrees (Counter-Clockwise)...")
    # PIL ROTATE_90 is Counter-Clockwise
    rotated_image = image.transpose(Image.Transpose.ROTATE_90)
    print(f"New Size: {rotated_image.size} (Width x Height)")
    rotated_image.save(output_path)
    print(f"Saved rotated image to: {os.path.abspath(output_path)}")
    
    print("\nVerification Logic:")
    print("- Original: Vertical strip (Top -> Bottom)")
    print("- Rotation: 90 degrees Counter-Clockwise")
    print("- Result: Horizontal strip (Left -> Right)")
    print("- This is the CORRECT direction for standard OCR models to read sequence.")
else:
    print("Image is already horizontal. No rotation needed.")
