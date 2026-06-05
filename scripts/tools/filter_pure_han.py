#!/usr/bin/env python3
import os
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
PRONUNCIATION_CSV = os.path.join(DATA_DIR, "all_chars_pronunciation.csv")
NOM_ALL_CSV = os.path.join(DATA_DIR, "chu_nom_all.csv")
HAN_OUTPUT_CSV = os.path.join(DATA_DIR, "han_characters_only.csv")
NOM_OUTPUT_CSV = os.path.join(DATA_DIR, "vietnamese_nom_only.csv")

def filter_nom_pure():
    if not os.path.exists(PRONUNCIATION_CSV):
        print(f"Error: {PRONUNCIATION_CSV} not found.")
        return
    if not os.path.exists(NOM_ALL_CSV):
        print(f"Error: {NOM_ALL_CSV} not found.")
        return

    print("📖 Đang tải danh sách chữ Nôm thuần Việt (4,232 chữ)...")
    df_nom_ref = pd.read_csv(NOM_ALL_CSV)
    nom_set = set(df_nom_ref['Chữ_Nôm'].dropna().astype(str))

    print("📖 Đang đọc dữ liệu từ tệp all_chars_pronunciation.csv...")
    df = pd.read_csv(PRONUNCIATION_CSV)
    
    han_records = []
    nom_records = []
    
    print("⚙️ Đang phân loại ký tự...")
    for _, row in df.iterrows():
        char = str(row['Phồn_thể'])
        if not char or pd.isna(row['Phồn_thể']):
            continue
            
        # Classify based on whether it is in the 100% native Nom set
        if char in nom_set:
            nom_records.append(row)
        else:
            han_records.append(row)

    # Save Standard Chinese Characters
    df_han = pd.DataFrame(han_records)
    df_han.to_csv(HAN_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 Thành công! Đã tách {len(df_han):,} chữ Hán tiêu chuẩn vào: {HAN_OUTPUT_CSV}")

    # Save Native Nom Characters
    df_nom = pd.DataFrame(nom_records)
    df_nom.to_csv(NOM_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"🎉 Thành công! Đã tách {len(df_nom):,} chữ Nôm thuần Việt 100% vào: {NOM_OUTPUT_CSV}")

if __name__ == "__main__":
    filter_nom_pure()
