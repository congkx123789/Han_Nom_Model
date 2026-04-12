import torch
import torchvision.models as models
import os

def verify_weights(path):
    if not os.path.exists(path):
        print(f"Error: File not found at {path}")
        return
    
    print(f"Loading weights from {path}...")
    try:
        # Load the state dict
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        print(f"Successfully loaded weights! Number of keys: {len(state_dict)}")
        
        # Load into model to be sure
        model = models.mobilenet_v3_small()
        model.load_state_dict(state_dict)
        print("Model state dict loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")

if __name__ == "__main__":
    weights_path = "/home/alida/Documents/Cursor/Han_Nom_Model/models/mobilenet_v3_small.pth"
    verify_weights(weights_path)
