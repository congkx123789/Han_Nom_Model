#!/usr/bin/env python3
import os
import requests
import zipfile
import io
import re
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
PRONUNCIATION_CSV = os.path.join(DATA_DIR, "all_chars_pronunciation.csv")
MISSING_CSV = os.path.join(DATA_DIR, "missing_pinyin.csv")

UNIHAN_ZIP_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"

def is_chinese_char(c):
    return '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\U00020000' <= c <= '\U0002a6df'

def extract_counterpart(definition):
    if not isinstance(definition, str):
        return None
    
    # Look for parenthesized expressions like:
    # (incorrect form of 功)
    # (same as 功)
    # (variant of 功)
    # (simplified form of 功)
    match = re.search(r'\((?:incorrect|simplified|same as|variant|vulgar|non-standard|abbreviated|archaic|ancient|interchangeable)(?:\s+form)?\s+of\s+([^)]+)\)', definition, re.IGNORECASE)
    if match:
        text = match.group(1)
        # Extract Chinese characters from the text
        chars = [c for c in text if is_chinese_char(c)]
        if chars:
            return chars[0]
            
    # Also look for "same as 功" or similar without parentheses at the start
    match2 = re.search(r'^(?:same as|variant of|vulgar form of)\s+([^;,.]+)', definition, re.IGNORECASE)
    if match2:
        text = match2.group(1)
        chars = [c for c in text if is_chinese_char(c)]
        if chars:
            return chars[0]
            
    return None

def main():
    if not os.path.exists(PRONUNCIATION_CSV):
        print(f"❌ Không tìm thấy {PRONUNCIATION_CSV}.")
        return

    print("📥 Đang tải cơ sở dữ liệu Unihan...")
    r = requests.get(UNIHAN_ZIP_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("✅ Đã tải xong Unihan zip.")

    # Dictionary to hold mappings from character -> standard character found in definition
    def_mappings = {}

    print("⚙️ Đang phân tích kDefinition từ Unihan_Readings.txt...")
    with z.open('Unihan_Readings.txt') as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if line.startswith("#") or not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3 and parts[1] == 'kDefinition':
                code_point = parts[0]
                definition = parts[2]
                
                try:
                    char = chr(int(code_point[2:], 16))
                    counterpart = extract_counterpart(definition)
                    if counterpart:
                        def_mappings[char] = counterpart
                except ValueError:
                    continue

    print(f"✅ Đã tìm thấy {len(def_mappings):,} ánh xạ dị thể từ định nghĩa tiếng Anh.")

    print("📖 Đang tải bảng phiên âm hiện tại...")
    df = pd.read_csv(PRONUNCIATION_CSV)
    
    pinyin_dict = df.set_index('Phồn_thể')['Pinyin'].dropna().to_dict()
    for _, row in df.iterrows():
        pinyin_dict[str(row['Giản_thể'])] = str(row['Pinyin'])

    filled_count = 0
    filled_details = []
    
    for idx, row in df.iterrows():
        pyp = row['Pinyin']
        if pd.isna(pyp) or str(pyp).strip() == "":
            trad = str(row['Phồn_thể'])
            
            std_char = def_mappings.get(trad)
            if std_char:
                std_pinyin = pinyin_dict.get(std_char, "")
                if pd.notna(std_pinyin) and str(std_pinyin).strip() != "":
                    df.at[idx, 'Pinyin'] = std_pinyin
                    filled_count += 1
                    filled_details.append(f"{trad} -> {std_char} ({std_pinyin})")

    # Save the updated database
    df.to_csv(PRONUNCIATION_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 Đã điền thêm Pinyin cho {filled_count:,} dị thể dựa trên định nghĩa!")
    if filled_details:
        print("Chi tiết một số chữ đã điền:")
        for detail in filled_details[:10]:
            print(f"  • {detail}")

    # Re-generate the missing pinyin report
    df['Pinyin'] = df['Pinyin'].fillna('').astype(str).str.strip()
    missing = df[df['Pinyin'] == '']
    missing.to_csv(MISSING_CSV, index=False, encoding='utf-8-sig')
    print(f"📁 Đã cập nhật tệp thiếu Pinyin tại: {MISSING_CSV}")
    print(f"📊 Số hàng thiếu Pinyin còn lại: {len(missing):,}")

if __name__ == "__main__":
    main()
