from PIL import Image
import numpy as np
import sys

def inspect_image(image_path):
    try:
        img = Image.open(image_path)
        print(f"Image format: {img.format}")
        print(f"Image size: {img.size}")
        print(f"Image mode: {img.mode}")
        
        data = np.array(img)
        print(f"Min pixel value: {data.min()}")
        print(f"Max pixel value: {data.max()}")
        print(f"Mean pixel value: {data.mean()}")
        
        unique_values = np.unique(data)
        print(f"Number of unique pixel values: {len(unique_values)}")
        if len(unique_values) < 10:
            print(f"Unique values: {unique_values}")
            
        if data.min() == 255 and data.max() == 255:
            print("CONCLUSION: The image is completely WHITE (blank).")
        elif data.min() == 0 and data.max() == 0:
            print("CONCLUSION: The image is completely BLACK.")
        else:
            print("CONCLUSION: The image contains some content.")
            
    except Exception as e:
        print(f"Error opening image: {e}")

if __name__ == "__main__":
    path = "/home/alida/Documents/Cursor/Han_Nom_Model/dataset/99D1/99D1_0.png"
    inspect_image(path)
