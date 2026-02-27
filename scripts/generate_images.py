import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
import random
import numpy as np

def add_noise(image):
    """Thêm nhiễu Gaussian vào ảnh."""
    np_image = np.array(image).astype(np.float32) # Convert to float to avoid wrapping
    noise = np.random.normal(0, 5, np_image.shape) # Reduced sigma to 5, keep as float
    noisy_image = np.clip(np_image + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

def apply_augmentations(image):
    """Áp dụng các phép biến đổi để ảnh trông thực tế hơn."""
    # 1. Xoay ngẫu nhiên (góc nhỏ)
    angle = random.uniform(-5, 5)
    image = image.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=255)
    
    # 2. Thêm nhiễu
    image = add_noise(image)
    
    # 3. Làm mờ nhẹ
    # image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 0.5)))
    
    return image

def generate_char_image(char, font_path, size=64, output_dir="dataset"):
    """Tạo ảnh cho một chữ Hán/Nôm."""
    # Tạo ảnh trắng
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    
    try:
        # Load font
        font = ImageFont.truetype(font_path, int(size * 0.8))
        
        # Check if character is supported by font
        # getmask returns a mask image, getbbox returns the bounding box of non-zero regions
        mask = font.getmask(char)
        if not mask.getbbox():
            print(f"Skipping {char} (U+{ord(char):04X}): Not supported by font.")
            return

        # Tính toán vị trí để căn giữa
        # Dùng font.getbbox để lấy kích thước chữ
        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        position = ((size - w) // 2, (size - h) // 2 - bbox[1])
        
        # Vẽ chữ (màu đen)
        draw.text(position, char, font=font, fill=0)
        
        # Áp dụng augmentation
        image = apply_augmentations(image)
        
        # Lưu ảnh
        char_hex = hex(ord(char))[2:].upper()
        char_dir = os.path.join(output_dir, char_hex)
        os.makedirs(char_dir, exist_ok=True)
        
        # Lưu nhiều bản với các augmentation khác nhau
        for i in range(5): # Tạo 5 biến thể cho mỗi chữ
            aug_image = apply_augmentations(image)
            aug_image.save(os.path.join(char_dir, f"{char_hex}_{i}.png"))
            
    except Exception as e:
        print(f"Lỗi khi tạo ảnh cho chữ {char}: {e}")

def main():
    # Đường dẫn font (Người dùng đã cung cấp font này)
    font_path = "/home/alida/Documents/Cursor/Han_Nom_Model/fonts/NomNaTong-Regular.ttf"
    
    if not os.path.exists(font_path):
        print(f"CẢNH BÁO: Không tìm thấy font tại {font_path}")
        print("Vui lòng tải font chữ Nôm (ví dụ: NomNaTong-Regular.ttf) và để vào thư mục fonts/.")
        return

    # Load danh sách chữ từ CSV (lấy 100 chữ đầu để test)
    csv_path = "/home/alida/Documents/Cursor/Han_Nom_Model/data/Unihan_Vietnamese.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df = pd.read_csv(csv_path)
        all_chars = df['char'].dropna().unique()
        
        # Shuffle and pick 100 random characters to ensure we get some common ones
        # instead of just the first few (which might be rare Extension B chars)
        random.seed(42) # For reproducibility
        chars = random.sample(list(all_chars), min(100, len(all_chars)))
        
        print(f"Đang tạo ảnh cho {len(chars)} chữ ngẫu nhiên từ {len(all_chars)} ký tự...")
        for char in chars:
            generate_char_image(char, font_path)
        print("Hoàn thành!")
    else:
        print(f"Không tìm thấy file dữ liệu: {csv_path}")

if __name__ == "__main__":
    main()
