import torch
import torchvision.models as models
import timm
import os

def download_mobilenet():
    print("Downloading MobileNetV3-Small backbone...")
    # This will download the weights to the default torch cache directory
    backbone = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    print("MobileNetV3-Small backbone downloaded successfully.")

def download_paddleocr():
    print("Downloading PaddleOCR models (PP-OCRv4)...")
    from paddleocr import PaddleOCR
    # This will trigger the download of detection, recognition, and classification models
    # Fixed arguments for version 3.3.2
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    print("PaddleOCR models downloaded successfully.")

def download_vim():
    print("Downloading Vision Mamba (Vim-Tiny) weights...")
    import sys
    # Add Vim repo to path to allow timm to find the registered models
    vim_path = "/home/alida/Documents/Cursor/Han_Nom_Model/models/Vim/vim"
    if vim_path not in sys.path:
        sys.path.append(vim_path)
    
    # Try multiple possible names
    model_names = [
        'vim_tiny_patch16_224_bimamba', # User provided
        'vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2', # From source
        'vim_tiny_patch16_224' # Generic
    ]
    
    import models_mamba
    for name in model_names:
        try:
            print(f"Attempting to load model: {name}")
            model = timm.create_model(name, pretrained=True)
            print(f"Vision Mamba ({name}) weights downloaded successfully.")
            return
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    print("Could not download Vim weights automatically. Please refer to the Hugging Face links in models/Vim/README.md.")

if __name__ == "__main__":
    try:
        download_mobilenet()
        download_paddleocr()
        download_vim()
    except Exception as e:
        print(f"Error in main download loop: {e}")
