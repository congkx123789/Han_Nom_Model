#!/usr/bin/env python3
import os
import requests
import zipfile
import io
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
MAPPING_PATH = os.path.join(DATA_DIR, "all_chars_mapping.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "all_chars_pronunciation.csv")

UNIHAN_ZIP_URL = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"

def parse_unihan(zip_file, filename, field_name):
    data = []
    print(f"   • Đang trích xuất trường '{field_name}' từ {filename}...")
    with zip_file.open(filename) as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if line.startswith("#") or not line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 3 and parts[1] == field_name:
                code_point = parts[0]
                try:
                    char = chr(int(code_point[2:], 16))
                    value = parts[2]
                    data.append({'char': char, field_name: value})
                except ValueError:
                    continue
    return pd.DataFrame(data)

def main():
    if not os.path.exists(MAPPING_PATH):
        print(f"❌ Không tìm thấy {MAPPING_PATH}. Vui lòng chạy sinh tệp ánh xạ trước.")
        return

    print("📥 Đang tải cơ sở dữ liệu Unihan chính thức từ Unicode Consortium...")
    r = requests.get(UNIHAN_ZIP_URL)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("✅ Đã tải xong Unihan zip.")

    # Trích xuất phiên âm Hán-Việt và Pinyin (kMandarin)
    df_viet = parse_unihan(z, 'Unihan_Readings.txt', 'kVietnamese')
    df_mandarin = parse_unihan(z, 'Unihan_Readings.txt', 'kMandarin')

    print("⚙️ Đang gộp dữ liệu Unihan...")
    # Merge on char
    unihan_df = pd.merge(df_viet, df_mandarin, on='char', how='outer')
    
    # Load local dictionary data for back-up
    # Load Thieu Chuu for additional Han-Viet pronunciations
    thieu_chuu_path = os.path.join(DATA_DIR, "Thieu_Chuu_Dictionary.csv")
    tc_hv = {}
    if os.path.exists(thieu_chuu_path):
        df_tc = pd.read_csv(thieu_chuu_path)
        for _, row in df_tc.iterrows():
            if pd.notna(row['char']) and pd.notna(row['pronunciation']):
                tc_hv[str(row['char'])] = str(row['pronunciation'])

    # Load CVDICT for additional Pinyin pronunciations
    cvdict_path = os.path.join(DATA_DIR, "CVDICT_Trung_Viet.csv")
    cv_pinyin = {}
    if os.path.exists(cvdict_path):
        df_cv = pd.read_csv(cvdict_path)
        for _, row in df_cv.iterrows():
            trad = str(row.get('Phồn_thể', ''))
            pinyin = str(row.get('Pinyin', ''))
            if len(trad) == 1 and pinyin:
                cv_pinyin[trad] = pinyin.split(' ')[0].lower()

    # Load our base characters mapping
    df_map = pd.read_csv(MAPPING_PATH)

    # Convert Unihan to dicts for fast lookup
    unihan_viet_dict = unihan_df.set_index('char')['kVietnamese'].dropna().to_dict()
    unihan_mand_dict = unihan_df.set_index('char')['kMandarin'].dropna().to_dict()

    records = []
    print("⚙️ Đang đồng bộ hóa phiên âm và ánh xạ...")
    for _, row in df_map.iterrows():
        trad = str(row['Phồn_thể'])
        simp = str(row['Giản_thể'])

        # Han-Viet lookup: Priority Unihan -> Thiều Chửu
        hv_pron = unihan_viet_dict.get(trad, unihan_viet_dict.get(simp, tc_hv.get(trad, tc_hv.get(simp, ""))))
        
        # Pinyin lookup: Priority Unihan kMandarin -> CVDICT character Pinyin
        pinyin_pron = unihan_mand_dict.get(trad, unihan_mand_dict.get(simp, cv_pinyin.get(trad, cv_pinyin.get(simp, ""))))

        records.append({
            "Phồn_thể": trad,
            "Giản_thể": simp,
            "Pinyin": pinyin_pron,
            "Phiên_âm_Hán_Việt": hv_pron
        })

    # Save output
    df_output = pd.DataFrame(records)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print(f"🎉 Hoàn thành! Đã lưu cơ sở dữ liệu phiên âm đầy đủ tại: {OUTPUT_CSV}")
    print(f"📊 Thống kê phiên âm sau khi gộp Unihan:")
    print(f"   • Pinyin count: {df_output['Pinyin'].notna().sum():,} / {len(df_output):,}")
    print(f"   • Han-Viet count: {df_output['Phiên_âm_Hán_Việt'].notna().sum():,} / {len(df_output):,}")

if __name__ == "__main__":
    main()
