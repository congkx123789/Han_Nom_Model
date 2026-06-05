#!/usr/bin/env python3
import os
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
CVDICT_PATH = os.path.join(DATA_DIR, "CVDICT_Trung_Viet.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "trad_to_simp.csv")

def extract_mappings():
    if not os.path.exists(CVDICT_PATH):
        print(f"Error: {CVDICT_PATH} not found.")
        return

    print("📖 Đang đọc từ điển CVDICT_Trung_Viet.csv...")
    df = pd.read_csv(CVDICT_PATH)
    
    mapping = {}
    
    for _, row in df.iterrows():
        trad = str(row.get('Phồn_thể', ''))
        simp = str(row.get('Giản_thể', ''))
        
        # Only compare if they have the same length
        if len(trad) == len(simp):
            for t_char, s_char in zip(trad, simp):
                # We only want to map characters that are actually different
                if t_char != s_char:
                    # Check if character is a Chinese character (CJK range)
                    if '\u4e00' <= t_char <= '\u9fff' or '\u3400' <= t_char <= '\u4dbf':
                        mapping[t_char] = s_char

    # Convert to DataFrame
    mapped_df = pd.DataFrame(list(mapping.items()), columns=['Phồn_thể', 'Giản_thể'])
    # Sort by Traditional character
    mapped_df = mapped_df.sort_values(by='Phồn_thể')
    
    # Save to CSV
    mapped_df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8-sig')
    print(f"✅ Đã trích xuất thành công {len(mapped_df)} cặp chữ Phồn thể -> Giản thể!")
    print(f"📁 Đã lưu tệp tại: {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_mappings()
