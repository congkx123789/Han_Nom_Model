#!/usr/bin/env python3
import os
import requests
import zipfile
import io
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
PRONUNCIATION_CSV = os.path.join(DATA_DIR, "all_chars_pronunciation.csv")
MISSING_CSV = os.path.join(DATA_DIR, "missing_pinyin.csv")

UNIHAN_ZIP_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"

def parse_code_point(cp_str):
    """Convert U+XXXX string to actual character"""
    cp_str = cp_str.strip().split('<')[0] # Remove source details like <kFenn
    if cp_str.startswith("U+"):
        try:
            return chr(int(cp_str[2:], 16))
        except ValueError:
            return None
    return None

def main():
    if not os.path.exists(PRONUNCIATION_CSV):
        print(f"❌ Không tìm thấy {PRONUNCIATION_CSV}. Vui lòng chạy sinh tệp phát âm trước.")
        return

    print("📥 Đang tải cơ sở dữ liệu Unihan...")
    r = requests.get(UNIHAN_ZIP_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("✅ Đã tải xong Unihan zip.")

    # Dictionary to hold mappings from variant character -> list of standard characters
    variant_to_standard = {}

    variant_fields = ['kSemanticVariant', 'kSpecializedSemanticVariant', 'kZVariant', 'kCompatibilityVariant']

    print("⚙️ Đang phân tích cú pháp các mối quan hệ dị thể từ Unihan_Variants.txt...")
    with z.open('Unihan_Variants.txt') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if line.startswith("#") or not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3 and parts[1] in variant_fields:
                source_cp = parts[0]
                relation = parts[1]
                target_cps = parts[2].split(' ') # can be space-separated list
                
                source_char = parse_code_point(source_cp)
                if not source_char:
                    continue
                    
                if source_char not in variant_to_standard:
                    variant_to_standard[source_char] = set()
                
                for target_cp in target_cps:
                    target_char = parse_code_point(target_cp)
                    if target_char:
                        variant_to_standard[source_char].add(target_char)

    print("📖 Đang tải bảng phiên âm hiện tại...")
    df = pd.read_csv(PRONUNCIATION_CSV)
    
    # Create index for fast lookup of Pinyin and Han-Viet by character
    pinyin_dict = df.set_index('Phồn_thể')['Pinyin'].dropna().to_dict()
    # Add simplified forms to lookup index as well
    for _, row in df.iterrows():
        pinyin_dict[str(row['Giản_thể'])] = str(row['Pinyin'])

    filled_count = 0
    print("⚙️ Bắt đầu điền Pinyin cho các dị thể...")
    
    for idx, row in df.iterrows():
        pyp = row['Pinyin']
        if pd.isna(pyp) or str(pyp).strip() == "":
            trad = str(row['Phồn_thể'])
            
            # Find standard characters related to this variant
            standards = variant_to_standard.get(trad, [])
            for std_char in standards:
                # Check if the standard character has a Pinyin in our database
                std_pinyin = pinyin_dict.get(std_char, "")
                if pd.notna(std_pinyin) and str(std_pinyin).strip() != "":
                    df.at[idx, 'Pinyin'] = std_pinyin
                    filled_count += 1
                    break # Use the first found Pinyin

    # Save the updated database
    df.to_csv(PRONUNCIATION_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 Đã điền thêm Pinyin cho {filled_count:,} dị thể!")
    
    # Re-generate the missing pinyin report
    df['Pinyin'] = df['Pinyin'].fillna('').astype(str).str.strip()
    missing = df[df['Pinyin'] == '']
    missing.to_csv(MISSING_CSV, index=False, encoding='utf-8-sig')
    print(f"📁 Đã cập nhật tệp thiếu Pinyin tại: {MISSING_CSV}")
    print(f"📊 Số hàng thiếu Pinyin còn lại: {len(missing):,}")

if __name__ == "__main__":
    main()
