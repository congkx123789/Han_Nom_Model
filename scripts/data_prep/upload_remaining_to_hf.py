import os
import sys
import socket
import shutil
import time
from huggingface_hub import HfApi

# Force IPv6 for network-only environment
orig_getaddrinfo = socket.getaddrinfo
def getaddrinfo_ipv6(host, port, family=0, type=0, proto=0, flags=0):
    return orig_getaddrinfo(host, port, socket.AF_INET6, type, proto, flags)
socket.getaddrinfo = getaddrinfo_ipv6

token = os.getenv("HF_TOKEN")
repo_id = 'Cong123779/Han_Nom_Dataset'
api = HfApi(token=token)

missing_files = [
    ('data/CVDICT_Trung_Viet.csv', 'data/CVDICT_Trung_Viet.csv'),
    ('data/Thieu_Chuu_Dictionary.csv', 'data/Thieu_Chuu_Dictionary.csv'),
    ('data/Unihan_Vietnamese.csv', 'data/Unihan_Vietnamese.csv'),
    ('data/all_chars_pronunciation.csv', 'data/all_chars_pronunciation.csv'),
    ('data/total_war_harvest.json', 'total_war_harvest.json'),
    ('data/crawler_state.db', 'crawler_state.db'),
    ('data/total_war_state.db', 'total_war_state.db'),
    ('data/mass_heritage_data.json', 'mass_heritage_data.json'),
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

print("="*60)
print("STARTING UPLOAD OF 19 MISSING DATA FILES TO HUGGING FACE")
print("="*60)

success_count = 0
fail_count = 0

for local, repo_path in missing_files:
    if not os.path.exists(local):
        print(f"SKIPPED (Not found local): {local}")
        fail_count += 1
        continue
        
    upload_file_path = local
    temp_copied = False
    
    # Handle DB files (copy to tmp to prevent DB lock issues during upload)
    if local.endswith('.db'):
        temp_path = f"/tmp/upload_{os.path.basename(local)}"
        print(f" -> Copying {local} to {temp_path} to prevent lock issues...")
        try:
            shutil.copy2(local, temp_path)
            upload_file_path = temp_path
            temp_copied = True
        except Exception as copy_err:
            print(f" -> Warning copying: {copy_err}. Attempting to upload original.")

    retries = 3
    success = False
    for attempt in range(1, retries + 1):
        print(f"Uploading {local} to Hugging Face {repo_path} (Attempt {attempt}/{retries})...")
        try:
            api.upload_file(
                path_or_fileobj=upload_file_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type='dataset'
            )
            print(f"✅ SUCCESS: {local}")
            success = True
            success_count += 1
            break
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(3)
                
    if temp_copied and os.path.exists(upload_file_path):
        os.remove(upload_file_path)
        
    if not success:
        print(f"🔴 PERMANENT FAIL: {local}")
        fail_count += 1

print("="*60)
print("UPLOAD SUMMARY:")
print(f" -> Successful uploads: {success_count}/19")
print(f" -> Failed uploads: {fail_count}/19")
print("="*60)
