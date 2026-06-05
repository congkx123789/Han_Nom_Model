#!/usr/bin/env python3
import os
import pandas as pd

# File Paths
DATA_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data"
MISSING_CSV = os.path.join(DATA_DIR, "missing_pinyin.csv")
NOM_OUTPUT_CSV = os.path.join(DATA_DIR, "chu_nom_all.csv")

def extract_nom_chars():
    if not os.path.exists(MISSING_CSV):
        print(f"Error: {MISSING_CSV} not found.")
        return

    print("📖 Đang đọc dữ liệu từ tệp missing_pinyin.csv...")
    df = pd.read_csv(MISSING_CSV)
    
    nom_records = []
    
    print("⚙️ Đang lọc chữ Nôm (Khối Extension B trở lên)...")
    for _, row in df.iterrows():
        char = str(row['Phồn_thể'])
        if not char or pd.isna(row['Phồn_thể']):
            continue
            
        cp = ord(char)
        # Extension B (0x20000 - 0x2A6DF) and above extensions contain native Nom characters
        if 0x20000 <= cp <= 0x2f7ff:
            nom_records.append({
                "Chữ_Nôm": char,
                "Phiên_âm_Hán_Việt": row['Phiên_âm_Hán_Việt']
            })

    # Save to CSV
    df_nom = pd.DataFrame(nom_records)
    # Remove duplicates if any (should be unique already)
    df_nom = df_nom.drop_duplicates(subset=['Chữ_Nôm'])
    # Sort by Nom character
    df_nom = df_nom.sort_values(by='Chữ_Nôm')
    
    df_nom.to_csv(NOM_OUTPUT_CSV, index=False, encoding='utf-8-sig')
    
    print(f"🎉 Hoàn thành! Đã trích xuất {len(df_nom):,} chữ Nôm thuần Việt.")
    print(f"📁 Tệp đã được lưu tại: {NOM_OUTPUT_CSV}")

if __name__ == "__main__":
    nom_records = extract_nom_chars()
