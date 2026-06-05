import os
import zipfile
import shutil
import urllib.request
import pandas as pd

# Cấu hình các bộ dữ liệu cần tải từ OPUS
DATASETS_TO_DOWNLOAD = {
    "TED2020": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/vi-zh.txt.zip",
    "ALT": "https://object.pouta.csc.fi/OPUS-ALT/v20191206/moses/vi-zh.txt.zip",
    "WikiMatrix": "https://object.pouta.csc.fi/OPUS-WikiMatrix/v1/moses/vi-zh.txt.zip",
    "OpenSubtitles": "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2016/moses/vi-zh.txt.zip",
    "NLLB": "https://object.pouta.csc.fi/OPUS-NLLB/v1/moses/vi-zh.txt.zip"
}

OUTPUT_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data/parallel_corpora"
TEMP_DIR = "/home/alida/Documents/Cursor/Han_Nom_Model/data/parallel_corpora/temp"

def download_progress(block_num, block_size, total_size):
    """Hiển thị phần trăm tải file."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        print(f"\r -> Đang tải: {percent:.1f}% ({downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)", end="")
    else:
        print(f"\r -> Đang tải: {downloaded / (1024*1024):.2f} MB", end="")

def process_corpus(name, url):
    print(f"\n==================================================")
    print(f"BẮT ĐẦU XỬ LÝ DATASET: {name}")
    print(f"==================================================")
    
    zip_path = os.path.join(OUTPUT_DIR, f"{name}.zip")
    extract_path = os.path.join(TEMP_DIR, name)
    
    # 1. Tải file zip
    try:
        print(f"Đang tải từ: {url}")
        urllib.request.urlretrieve(url, zip_path, download_progress)
        print(f"\n[OK] Đã tải xong {name}.zip")
    except Exception as e:
        print(f"\n[LỖI] Tải thất bại: {e}")
        return 0
        
    # 2. Giải nén
    try:
        print(f"Đang giải nén file...")
        os.makedirs(extract_path, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"[OK] Giải nén thành công.")
    except Exception as e:
        print(f"[LỖI] Giải nén thất bại: {e}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return 0

    # 3. Tìm file text moses (.zh và .vi)
    files = os.listdir(extract_path)
    zh_file = None
    vi_file = None
    
    for f in files:
        if f.endswith('.zh') or f.endswith('.zh_cn') or f.endswith('.zh_tw'):
            zh_file = os.path.join(extract_path, f)
        elif f.endswith('.vi'):
            vi_file = os.path.join(extract_path, f)
            
    if not zh_file or not vi_file:
        print(f"[LỖI] Không tìm thấy file song ngữ (.zh và .vi) trong thư mục giải nén.")
        # Dọn dẹp
        shutil.rmtree(extract_path, ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return 0
        
    # 4. Đọc dữ liệu và ghép cặp câu
    try:
        print(f"Đang phân tích cú pháp và ghép cặp câu...")
        with open(zh_file, 'r', encoding='utf-8') as f_zh:
            zh_lines = [line.strip() for line in f_zh]
        with open(vi_file, 'r', encoding='utf-8') as f_vi:
            vi_lines = [line.strip() for line in f_vi]
            
        print(f" -> Số câu tiếng Trung: {len(zh_lines)}")
        print(f" -> Số câu tiếng Việt: {len(vi_lines)}")
        
        min_len = min(len(zh_lines), len(vi_lines))
        if min_len == 0:
            print(f"[LỖI] File dữ liệu bị rỗng.")
            shutil.rmtree(extract_path, ignore_errors=True)
            os.remove(zip_path)
            return 0
            
        # Ghép cặp
        df = pd.DataFrame({
            'zh': zh_lines[:min_len],
            'vi': vi_lines[:min_len]
        })
        
        # Loại bỏ các dòng trống
        df = df.dropna()
        df = df[(df['zh'] != '') & (df['vi'] != '')]
        
        # Lưu thành CSV
        out_csv = os.path.join(OUTPUT_DIR, f"{name}_zh_vi.csv")
        df.to_csv(out_csv, index=False, encoding='utf-8')
        print(f"[OK] Đã lưu {len(df)} cặp câu sạch vào: {out_csv}")
        
        # Dọn dẹp file trung gian lập tức để tiết kiệm đĩa
        shutil.rmtree(extract_path, ignore_errors=True)
        os.remove(zip_path)
        return len(df)
        
    except Exception as e:
        print(f"[LỖI] Gặp sự cố khi đọc/lưu dữ liệu: {e}")
        shutil.rmtree(extract_path, ignore_errors=True)
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return 0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    total_loaded = 0
    results = {}
    
    for name, url in DATASETS_TO_DOWNLOAD.items():
        count = process_corpus(name, url)
        results[name] = count
        total_loaded += count
        
    # Xóa thư mục temp
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    print("\n" + "=" * 60)
    print("HOÀN THÀNH QUÁ TRÌNH TẢI VÀ XỬ LÝ DATASET DỊCH SONG NGỮ!")
    print("=" * 60)
    for name, count in results.items():
        print(f" - {name}: {count:,} cặp câu")
    print(f"TỔNG CỘNG: {total_loaded:,} cặp câu song ngữ đã sẵn sàng.")
    print(f"Tất cả file được lưu tại: {OUTPUT_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
