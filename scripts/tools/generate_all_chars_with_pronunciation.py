#!/usr/bin/env python3
import os
import pandas as pd
from collections import Counter

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
CVDICT_PATH = os.path.join(DATA_DIR, "CVDICT_Trung_Viet.csv")
THIEUCHUU_PATH = os.path.join(DATA_DIR, "Thieu_Chuu_Dictionary.csv")
UNIHAN_PATH = os.path.join(DATA_DIR, "Unihan_Vietnamese.csv")
MAPPING_PATH = os.path.join(DATA_DIR, "all_chars_mapping.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "all_chars_pronunciation.csv")

def merge_pronunciations():
    if not os.path.exists(MAPPING_PATH):
        print(f"Error: {MAPPING_PATH} not found. Please generate it first.")
        return

    print("📖 Đang tải bảng ánh xạ chữ đơn...")
    df_map = pd.read_csv(MAPPING_PATH)

    # 1. Load Han-Viet pronunciations
    hv_unihan = {}
    if os.path.exists(UNIHAN_PATH):
        print("📖 Đang tải âm Hán-Việt từ Unihan...")
        df_unihan = pd.read_csv(UNIHAN_PATH)
        for _, row in df_unihan.iterrows():
            char = str(row['char'])
            viet = str(row['kVietnamese'])
            if pd.notna(row['char']) and pd.notna(row['kVietnamese']):
                hv_unihan[char] = viet.strip()

    hv_thieuchuu = {}
    if os.path.exists(THIEUCHUU_PATH):
        print("📖 Đang tải âm Hán-Việt từ Thiệu Chửu...")
        df_tc = pd.read_csv(THIEUCHUU_PATH)
        for _, row in df_tc.iterrows():
            char = str(row['char'])
            pron = str(row['pronunciation'])
            if pd.notna(row['char']) and pd.notna(row['pronunciation']):
                hv_thieuchuu[char] = pron.strip()

    # 2. Extract Pinyin from CVDICT
    pinyin_data = {} # char -> Counter of pinyins
    if os.path.exists(CVDICT_PATH):
        print("📖 Đang phân tích âm Pinyin từ CVDICT...")
        df_cv = pd.read_csv(CVDICT_PATH)
        for _, row in df_cv.iterrows():
            trad = str(row.get('Phồn_thể', ''))
            pinyin_str = str(row.get('Pinyin', ''))
            
            if not trad or not pinyin_str or pd.isna(row['Phồn_thể']) or pd.isna(row['Pinyin']):
                continue
                
            # Split Pinyin into syllables
            syllables = [s.strip() for s in pinyin_str.split(' ') if s.strip()]
            
            # Align only if the number of characters matches the number of syllables
            if len(trad) == len(syllables):
                for char, pyl in zip(trad, syllables):
                    # Clean pinyin a bit (lowercase, keep tone number)
                    pyl_clean = pyl.lower()
                    if char not in pinyin_data:
                        pinyin_data[char] = Counter()
                    pinyin_data[char][pyl_clean] += 1

    # 3. Process merged records
    print("⚙️ Đang tích hợp phiên âm cho từng chữ...")
    records = []
    
    for _, row in df_map.iterrows():
        trad = str(row['Phồn_thể'])
        simp = str(row['Giản_thể'])
        
        # Determine Han-Viet pronunciation
        # Priority: Unihan -> Thiều Chửu -> N/A
        hv_pron = hv_unihan.get(trad, hv_thieuchuu.get(trad, ""))
        if not hv_pron:
            # Try lookup using Simplified character if Traditional didn't match
            hv_pron = hv_unihan.get(simp, hv_thieuchuu.get(simp, ""))
            
        # Determine Pinyin
        # Choose the most frequent Pinyin found, or check simplified character if not found
        pys = pinyin_data.get(trad, pinyin_data.get(simp, None))
        if pys:
            # Sort by frequency and take the most common one, or list top common ones
            most_common_pys = [item[0] for item in pys.most_common(2)] # up to 2 variations
            pinyin_pron = "/".join(most_common_pys)
        else:
            pinyin_pron = ""
            
        records.append({
            "Phồn_thể": trad,
            "Giản_thể": simp,
            "Pinyin": pinyin_pron,
            "Phiên_âm_Hán_Việt": hv_pron
        })

    # Save to CSV
    df_output = pd.DataFrame(records)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"🎉 Hoàn thành! Đã tích hợp phiên âm cho {len(df_output):,} chữ đơn.")
    print(f"📁 Tệp đã được lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    merge_pronunciations()
