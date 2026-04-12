import os
import sys
from huggingface_hub import HfApi, CommitOperationAdd
from tqdm import tqdm

def upload_dataset(start_idx_overall=0):
    api = HfApi()
    repo_id = "Cong123779/Han_Nom_Dataset"
    token = os.getenv("HF_TOKEN")

    dataset_parts = [
        {"local": "data/raw", "repo": "raw"},
        {"local": "data/synthetic_nom", "repo": "synthetic_nom"},
        {"local": "data/NomNaOCR", "repo": "NomNaOCR"},
        {"local": "data/ids", "repo": "ids"},
        {"local": "data/Thieu_Chuu_Dictionary.csv", "repo": "Thieu_Chuu_Dictionary.csv"},
        {"local": "data/Unihan_Vietnamese.csv", "repo": "Unihan_Vietnamese.csv"},
    ]

    print("Building full file list...")
    all_local_files = []
    for part in dataset_parts:
        if os.path.isfile(part["local"]):
            all_local_files.append((part["local"], part["repo"]))
        elif os.path.isdir(part["local"]):
            for root, dirs, files in os.walk(part["local"]):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, part["local"])
                    all_local_files.append((full_path, os.path.join(part["repo"], rel_path)))
        else:
            print(f"Warning: {part['local']} not found.")

    all_local_files.sort() # Consistent ordering
    print(f"Total files in dataset: {len(all_local_files)}")
    
    batch_size = 1000
    num_batches = (len(all_local_files) + batch_size - 1) // batch_size
    
    # Calculate starting batch if provided
    start_batch = (start_idx_overall // batch_size) + 1
    
    print(f"Starting/Resuming from Batch {start_batch}/{num_batches} (Index {start_idx_overall})...")

    for i in range(start_idx_overall, len(all_local_files), batch_size):
        batch = all_local_files[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"Uploading batch {batch_num}/{num_batches} ({len(batch)} files)...")
        
        operations = [
            CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=local_path)
            for local_path, repo_path in batch
        ]
        
        api.create_commit(
            repo_id=repo_id,
            operations=operations,
            commit_message=f"Upload batch {batch_num}/{num_batches}",
            repo_type="dataset",
            token=token
        )
        print(f"Batch {batch_num} committed successfully.")

if __name__ == "__main__":
    start_idx = 0
    if len(sys.argv) > 1:
        # If user provides a batch number, convert it to start index
        start_batch = int(sys.argv[1])
        start_idx = (start_batch - 1) * 1000
    upload_dataset(start_idx)
