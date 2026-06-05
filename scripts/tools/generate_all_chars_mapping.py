#!/usr/bin/env python3
import os
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
CVDICT_PATH = os.path.join(DATA_DIR, "CVDICT_Trung_Viet.csv")
THIEUCHUU_PATH = os.path.join(DATA_DIR, "Thieu_Chuu_Dictionary.csv")
UNIHAN_PATH = os.path.join(DATA_DIR, "Unihan_Vietnamese.csv")
OPENCC_PATH = os.path.join(DATA_DIR, "trad_to_simp_all.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "all_chars_mapping.csv")

def is_chinese_char(c):
    return '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' or '\U00020000' <= c <= '\U0002a6df'

def generate_mapping():
    print("📖 Đang tải các bộ từ điển để thu thập ký tự...")
    
    unique_chars = set()
    
    # 1. Collect from Unihan
    if os.path.exists(UNIHAN_PATH):
        df_unihan = pd.read_csv(UNIHAN_PATH)
        for val in df_unihan['char'].dropna():
            for c in str(val):
                if is_chinese_char(c):
                    unique_chars.add(c)
        print(f"   • Thu thập từ Unihan. Số lượng ký tự tích lũy: {len(unique_chars):,}")

    # 2. Collect from Thieu Chuu
    if os.path.exists(THIEUCHUU_PATH):
        df_tc = pd.read_csv(THIEUCHUU_PATH)
        for val in df_tc['char'].dropna():
            for c in str(val):
                if is_chinese_char(c):
                    unique_chars.add(c)
        print(f"   • Thu thập từ Thiệu Chửu. Số lượng ký tự tích lũy: {len(unique_chars):,}")

    # 3. Collect from CVDICT
    if os.path.exists(CVDICT_PATH):
        df_cv = pd.read_csv(CVDICT_PATH)
        for col in ['Phồn_thể', 'Giản_thể']:
            for val in df_cv[col].dropna():
                for c in str(val):
                    if is_chinese_char(c):
                        unique_chars.add(c)
        print(f"   • Thu thập từ CVDICT. Số lượng ký tự tích lũy: {len(unique_chars):,}")

    # 4. Load OpenCC mapping
    mapping = {}
    if os.path.exists(OPENCC_PATH):
        df_opencc = pd.read_csv(OPENCC_PATH)
        for _, row in df_opencc.iterrows():
            mapping[str(row['Phồn_thể'])] = str(row['Giản_thể'])
        print(f"   • Đã tải {len(mapping):,} quy tắc chuyển đổi Phồn -> Giản của OpenCC.")
    else:
        print("⚠️ Cảnh báo: Không tìm thấy tệp OpenCC trad_to_simp_all.csv. Sẽ chỉ sử dụng cơ chế đối sánh 1-1 giống nhau.")

    # 5. Build full mapping table
    print("⚙️ Đang xử lý bảng đối chiếu toàn bộ ký tự...")
    records = []
    for c in sorted(list(unique_chars)):
        # If the character has a simplified version, map to it. Otherwise, maps to itself.
        simp_version = mapping.get(c, c)
        records.append({
            "Phồn_thể": c,
            "Giản_thể": simp_version
        })

    # Save to CSV
    df_output = pd.DataFrame(records)
    df_output.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"🎉 Hoàn thành! Đã tạo bảng ánh xạ cho {len(df_output):,} ký tự đơn.")
    print(f"📁 Tệp đã được lưu tại: {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_mapping()
