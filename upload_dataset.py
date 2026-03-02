import os
from huggingface_hub import HfApi

def upload_dataset():
    api = HfApi()
    repo_id = "Cong123779/Han_Nom_Dataset"
    
    print(f"Creating repository {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    except Exception as e:
        print(f"Repo creation notice: {e}")
        
    local_dir = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
    
    print(f"Uploading files from {local_dir} to {repo_id} using upload_large_folder...")
    
    api.upload_large_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Uploading huge dataset via large_folder API",
    )
    
    print("Upload completed successfully!")

if __name__ == "__main__":
    upload_dataset()
