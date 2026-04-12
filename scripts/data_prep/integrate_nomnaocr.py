import os
import subprocess
import shutil

def clone_nomnaocr(target_dir):
    """Clone the NomNaOCR repository if it doesn't exist."""
    repo_url = "https://github.com/ds4v/NomNaOCR.git"
    if not os.path.exists(target_dir):
        print(f"Cloning NomNaOCR from {repo_url}...")
        subprocess.run(["git", "clone", "--depth", "1", repo_url, target_dir], check=True)
    else:
        print(f"NomNaOCR already exists at {target_dir}")

def process_nomnaocr(dataset_dir, output_dir):
    """
    Process the NomNaOCR dataset to extract Truyen Kieu data.
    Note: This is a placeholder for the actual processing logic, 
    as the exact structure of the repo needs to be inspected.
    """
    print(f"Processing NomNaOCR dataset from {dataset_dir}...")
    
    # Example: Look for 'Kieu' or 'TruyenKieu' folders
    # This will be refined once the repo is cloned and inspected.
    kieu_found = False
    for root, dirs, files in os.walk(dataset_dir):
        if 'Kieu' in root or 'TruyenKieu' in root:
            kieu_found = True
            print(f"Found Truyen Kieu data at: {root}")
            # Logic to copy/process images and labels would go here
            
    if not kieu_found:
        print("Warning: Could not find 'Kieu' or 'TruyenKieu' in the dataset.")

def main():
    base_dir = "/home/alida/Documents/Cursor/Han_Nom_Model"
    nomnaocr_dir = os.path.join(base_dir, "data", "NomNaOCR_Repo")
    processed_dir = os.path.join(base_dir, "data", "NomNaOCR_Processed")
    
    os.makedirs(os.path.dirname(nomnaocr_dir), exist_ok=True)
    
    try:
        clone_nomnaocr(nomnaocr_dir)
        process_nomnaocr(nomnaocr_dir, processed_dir)
    except Exception as e:
        print(f"Error integrating NomNaOCR: {e}")

if __name__ == "__main__":
    main()
