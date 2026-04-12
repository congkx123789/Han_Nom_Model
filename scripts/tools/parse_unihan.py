import pandas as pd
import requests
import zipfile
import io
import os

# 1. Tải và giải nén trong RAM (không cần lưu file zip)
url = "https://www.unicode.org/Public/UCD/latest/ucd/Unihan.zip"
print("Đang tải Unihan Database...")
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))

# 2. Hàm đọc file Unihan
def parse_unihan(zip_file, filename, field_name):
    data = []
    with zip_file.open(filename) as f:
        for line in f:
            line = line.decode('utf-8').strip()
            if line.startswith("#") or not line: continue
            
            # Cấu trúc: Mã_Unicode | Loại_Dữ_Liệu | Giá_Trị
            parts = line.split('\t')
            if len(parts) >= 3 and parts[1] == field_name:
                code_point = parts[0] # Ví dụ: U+4E00
                try:
                    char = chr(int(code_point[2:], 16)) # Chuyển U+4E00 -> chữ '一'
                    value = parts[2]
                    data.append({'hex': code_point, 'char': char, field_name: value})
                except ValueError:
                    continue
    return pd.DataFrame(data)

# 3. Trích xuất Âm Hán Việt
print("Đang trích xuất âm Hán Việt...")
df_viet = parse_unihan(z, 'Unihan_Readings.txt', 'kVietnamese')

# 4. Trích xuất Nghĩa tiếng Anh (để tham khảo)
print("Đang trích xuất nghĩa (Definition)...")
df_def = parse_unihan(z, 'Unihan_Readings.txt', 'kDefinition')

# 5. Gộp lại thành bảng từ điển
print("Đang gộp dữ liệu...")
df_final = pd.merge(df_viet, df_def, on=['hex', 'char'], how='outer')

# 6. Xem thử kết quả
print(f"Tổng số chữ Hán có âm Việt: {len(df_viet)}")
print(f"Tổng số chữ Hán có nghĩa: {len(df_def)}")
print(f"Tổng số chữ Hán sau khi gộp: {len(df_final)}")
print(df_final.head())

# Lưu ra CSV để dùng sau này
output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Unihan_Vietnamese.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_final.to_csv(output_path, index=False)
print(f"Đã lưu kết quả tại: {output_path}")
