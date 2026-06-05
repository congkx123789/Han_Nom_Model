#!/usr/bin/env python3
import os
import urllib.request
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
OUTPUT_CSV = os.path.join(DATA_DIR, "trad_to_simp_all.csv")

# OpenCC Raw URL for Traditional to Simplified character mapping
OPENCC_TS_CHAR_URL = "https://raw.githubusercontent.com/BYVoid/OpenCC/master/data/dictionary/TSCharacters.txt"

def download_and_parse():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"📥 Đang tải cơ sở dữ liệu từ OpenCC...")
    print(f"🔗 URL: {OPENCC_TS_CHAR_URL}")
    
    try:
        # Download the file
        with urllib.request.urlopen(OPENCC_TS_CHAR_URL) as response:
            content = response.read().decode('utf-8')
        
        print("✅ Đã tải xong! Đang phân tích cú pháp...")
        
        records = []
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # OpenCC dictionary files are tab-separated
            parts = line.split('\t')
            if len(parts) >= 2:
                trad_char = parts[0]
                # Simplified characters can be space-separated if there are multiple variations
                simp_chars = parts[1].split(' ')
                # Typically we want the primary simplified character (the first one)
                primary_simp = simp_chars[0]
                
                # Check for other variations
                other_variants = ", ".join(simp_chars[1:]) if len(simp_chars) > 1 else ""
                
                records.append({
                    "Phồn_thể": trad_char,
                    "Giản_thể": primary_simp,
                    "Biến_thể_khác": other_variants
                })
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
        
        print(f"🎉 Hoàn thành! Đã trích xuất {len(df):,} ký tự đơn Phồn thể -> Giản thể.")
        print(f"📁 Tệp đã được lưu tại: {OUTPUT_CSV}")
        
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi: {e}")

if __name__ == "__main__":
    download_and_parse()
