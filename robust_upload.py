import os
import sys
import socket
import shutil
import time

# Force IPv6 to avoid Network is unreachable errors (this system only has IPv6 connectivity)
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv6(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET6, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv6

from huggingface_hub import HfApi

# Output logging helper
log_file_path = "/home/alida/Documents/Cursor/Han_Nom_Model/upload_progress.log"
with open(log_file_path, "w") as f:
    f.write("=== Robust Upload Log V2.1 ===\n")

def log(message):
    print(message)
    with open(log_file_path, "a") as f:
        f.write(message + "\n")

api = HfApi()

repo_id = 'Cong123779/Han_Nom_Dataset'
try:
    api.create_repo(repo_id=repo_id, repo_type='dataset', exist_ok=True)
    log(f"Using repo: {repo_id}")
except Exception as e:
    log(f"Failed to verify repo: {e}")
    sys.exit(1)

# Prioritize volume_metadata.json, junda_char_freq.txt, and labels.csv first!
files = [
    # Metadata and raw files first as requested
    ('data/raw/volume_metadata.json', 'data/raw/volume_metadata.json'),
    ('data/raw/junda_char_freq.txt', 'data/raw/junda_char_freq.txt'),
    ('data/raw/labels.csv', 'data/raw/labels.csv'),
    # Dictionary CSVs
    ('data/CVDICT_Trung_Viet.csv', 'data/CVDICT_Trung_Viet.csv'),
    ('data/Thieu_Chuu_Dictionary.csv', 'data/Thieu_Chuu_Dictionary.csv'),
    ('data/Unihan_Vietnamese.csv', 'data/Unihan_Vietnamese.csv'),
    ('data/all_chars_pronunciation.csv', 'data/all_chars_pronunciation.csv'),
    # Crawled files
    ('data/cohoc_heritage_data.json', 'cohoc_heritage_data.json'),
    ('data/total_war_harvest.json', 'total_war_harvest.json'),
    ('data/crawler.log', 'crawler.log'),
    ('data/crawler_state.db', 'crawler_state.db'),
    ('data/total_war_state.db', 'total_war_state.db'),
    ('data/mass_heritage_data.json', 'mass_heritage_data.json'),
    # Other files
    ('data/approx_hv_by_radical.csv', 'data/approx_hv_by_radical.csv'),
    ('data/approx_hv_triple_model.csv', 'data/approx_hv_triple_model.csv'),
    ('data/approx_hv_updated.csv', 'data/approx_hv_updated.csv'),
    ('data/chu_nom_all.csv', 'data/chu_nom_all.csv'),
    ('data/han_characters_only.csv', 'data/han_characters_only.csv'),
    ('data/manual_corrections.csv', 'data/manual_corrections.csv'),
    ('data/missing_hv_candidates.csv', 'data/missing_hv_candidates.csv'),
    ('data/missing_pinyin.csv', 'data/missing_pinyin.csv'),
    ('data/polyphonic_master_missing.csv', 'data/polyphonic_master_missing.csv'),
    ('data/polyphonic_omitted_candidates.csv', 'data/polyphonic_omitted_candidates.csv'),
    ('data/vietnamese_nom_only.csv', 'data/vietnamese_nom_only.csv')
]

log("Starting upload of all files...")

for local, repo_path in files:
    if not os.path.exists(local):
        log(f"NOT FOUND: {local}")
        continue
    
    # Handle DB files separately to avoid lock issues
    upload_file_path = local
    temp_copied = False
    if local.endswith('.db'):
        temp_path = f"/tmp/{os.path.basename(local)}"
        log(f"Copying {local} to {temp_path} to bypass locks...")
        try:
            shutil.copy2(local, temp_path)
            upload_file_path = temp_path
            temp_copied = True
        except Exception as copy_err:
            log(f"Warning: Failed to copy {local}: {copy_err}. Proceeding with original file.")

    # Upload with retries
    retries = 3
    success = False
    for attempt in range(1, retries + 1):
        log(f"Uploading {local} to {repo_path} (Attempt {attempt}/{retries})...")
        try:
            api.upload_file(
                path_or_fileobj=upload_file_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type='dataset'
            )
            log(f"SUCCESS: {local}")
            success = True
            break
        except Exception as e:
            log(f"Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(5)
            
    if temp_copied and os.path.exists(upload_file_path):
        os.remove(upload_file_path)
        
    if not success:
        log(f"FAILED permanently: {local}")

log("Robust upload script completed!")
