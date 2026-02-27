import pandas as pd

# 1. Đọc file CSV
try:
    df = pd.read_csv('hanviet.csv')
    print("Successfully loaded hanviet.csv")
    print(f"Total entries: {len(df)}")
    print("Columns:", df.columns.tolist())
    print("-" * 30)
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# 2. Hàm tra cứu đơn giản
def tra_cuu_han_viet(ky_tu_han):
    ket_qua = df[df['char'] == ky_tu_han]
    if not ket_qua.empty:
        # Trả về danh sách các âm đọc tìm thấy
        return ket_qua[['hanviet', 'pinyin']].to_dict('records')
    else:
        return "Không tìm thấy trong từ điển này"

# 3. Test thử với chữ '中' (Có 2 âm đọc: Trung/Trúng)
print(f"Tra cứu chữ 中: {tra_cuu_han_viet('中')}")

# 4. Test thử với chữ '降' (Giáng/Hàng)
print(f"Tra cứu chữ 降: {tra_cuu_han_viet('降')}")

# 5. Test thử với chữ '行' (Hành/Hạnh/Hàng)
print(f"Tra cứu chữ 行: {tra_cuu_han_viet('行')}")
