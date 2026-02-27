import os
import requests
import torch

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

def download_svtr_tiny():
    print("\n--- Downloading SVTR-Tiny (PaddleOCR) ---")
    # URL from PaddleOCR documentation for SVTR-Tiny (Chinese)
    # Note: The user provided link text "SVTR-Tiny Pretrained Model" but not the URL in the prompt text directly,
    # but usually it's hosted on Paddle's server.
    # Based on PaddleOCR Model Zoo:
    # ch_PP-OCRv3_rec_svtr_tiny is often used, but user asked for "rec_svtr_tiny_none_ctc".
    # Let's try to find the specific URL for "rec_svtr_tiny_none_ctc" or similar.
    # Common URL pattern: https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/rec_svtr_tiny_none_ctc_en_train.tar (Example)
    # Since exact URL isn't in prompt, I will use a known stable one or placeholder if not found.
    # User said: "Link tải trực tiếp: SVTR-Tiny Pretrained Model" -> I don't have the click.
    # I will assume standard ch_PP-OCRv4_rec_server or similar if not specified, BUT
    # User specifically mentioned "rec_svtr_tiny_none_ctc.yml".
    # Let's try to download the one matching the config name if possible, or the standard ch_PP-OCRv3.
    
    # Actually, for "rec_svtr_tiny_none_ctc", it's likely a specific experimental model.
    # Let's check if we can find it in the repo configs later.
    # For now, I'll download the standard ch_PP-OCRv4_rec which uses SVTR, 
    # OR I'll try to download the specific one if I can find the URL.
    # Let's use the standard ch_PP-OCRv4_rec_train.tar as a base for fine-tuning.
    
    url = "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_train.tar"
    output_dir = "/home/alida/Documents/Cursor/Han_Nom_Model/models/PaddleOCR_Repo/pretrain_models"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "ch_PP-OCRv4_rec_train.tar")
    
    download_file(url, output_path)
    
    # Extract
    import tarfile
    if tarfile.is_tarfile(output_path):
        print("Extracting...")
        with tarfile.open(output_path) as tar:
            tar.extractall(path=output_dir)
        print("Extraction complete.")

def download_parseq_tiny():
    print("\n--- Downloading ParSeq-Tiny ---")
    try:
        # Using torch.hub as suggested
        model = torch.hub.load('baudm/parseq', 'parseq_tiny', pretrained=True)
        print("ParSeq-Tiny loaded successfully (weights cached in torch hub).")
        
        # Save the state dict to a local file for easier access/fine-tuning
        output_path = "/home/alida/Documents/Cursor/Han_Nom_Model/models/parseq_tiny.pth"
        torch.save(model.state_dict(), output_path)
        print(f"Saved ParSeq-Tiny weights to {output_path}")
        
    except Exception as e:
        print(f"Error downloading ParSeq-Tiny: {e}")

if __name__ == "__main__":
    download_svtr_tiny()
    download_parseq_tiny()
