import mobi
import os
import pandas as pd
from bs4 import BeautifulSoup
import html
import re

def convert_prc_to_csv(prc_path, output_csv_path):
    print(f"Đang trích xuất file PRC: {prc_path}...")
    temp_dir, filepath = mobi.extract(prc_path)
    print(f"File HTML đã trích xuất tại: {filepath}")

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    print("Đang phân tích HTML...")
    soup = BeautifulSoup(content, 'html.parser')

    data = []
    # Tìm tất cả các thẻ p
    p_tags = soup.find_all('p')
    
    print(f"Tìm thấy {len(p_tags)} thẻ <p>. Đang trích xuất dữ liệu...")

    for p in p_tags:
        # Một entry hợp lệ thường có thẻ font (chữ) và thẻ b (phiên âm)
        font_tag = p.find('font')
        b_tag = p.find('b')
        
        if font_tag and b_tag:
            char = font_tag.get_text().strip()
            pronunciation = b_tag.get_text().strip()
            
            # Bỏ qua các mục lục (index) nơi phiên âm chỉ là con số (số nét)
            if pronunciation.isdigit():
                continue
                
            # Bỏ qua các mục có chữ quá dài (thường là danh sách chữ trong mục lục)
            if len(char) > 5:
                continue
            # Chúng ta sẽ lấy toàn bộ text của p và loại bỏ phần chữ và phiên âm
            # Hoặc tốt hơn là lấy các sibling sau b_tag
            def_parts = []
            current = b_tag.next_sibling
            while current:
                if isinstance(current, str):
                    text = current.strip()
                    if text:
                        def_parts.append(text)
                else:
                    text = current.get_text().strip()
                    if text:
                        def_parts.append(text)
                current = current.next_sibling
            
            definition = " ".join(def_parts).strip()
            
            # Làm sạch definition (loại bỏ các ký tự thừa như (8n))
            definition = re.sub(r'^\s*\(\d+n\)\s*', '', definition)
            
            if char and pronunciation and definition:
                data.append({
                    'char': char,
                    'pronunciation': pronunciation,
                    'definition': definition
                })

    print(f"Đã trích xuất được {len(data)} từ.")
    
    if data:
        df = pd.DataFrame(data)
        # Loại bỏ các hàng trùng lặp nếu có
        df = df.drop_duplicates()
        df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
        print(f"Đã lưu kết quả tại: {output_csv_path}")
    else:
        print("Không tìm thấy dữ liệu để lưu.")

if __name__ == "__main__":
    prc_file = "/home/alida/Documents/Cursor/Han_Nom_Model/Han-Viet Tu-dien - Thieu Chuu.prc"
    output_csv = "/home/alida/Documents/Cursor/Han_Nom_Model/data/Thieu_Chuu_Dictionary.csv"
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    convert_prc_to_csv(prc_file, output_csv)
