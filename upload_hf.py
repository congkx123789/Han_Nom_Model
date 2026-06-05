from huggingface_hub import HfApi
import os

api = HfApi()

repo_id = 'Cong123779/han-nom-crawled-data'
try:
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
    print(f"Created/verified dataset repo {repo_id}")
except Exception as e:
    print(f"Repo creation issue: {e}")

files_to_upload = [
    'data/cohoc_heritage_data.json',
    'data/total_war_harvest.json',
    'data/crawler.log',
    'data/crawler_state.db',
    'data/total_war_state.db',
    'data/mass_heritage_data.json'
]

for file_path in files_to_upload:
    if os.path.exists(file_path):
        print(f"Uploading {file_path}...")
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path),
                repo_id=repo_id,
                repo_type='dataset'
            )
            print(f"Successfully uploaded {file_path}")
        except Exception as e:
            print(f"Failed to upload {file_path}: {e}")
    else:
        print(f"File not found: {file_path}")

print("Upload process finished!")
