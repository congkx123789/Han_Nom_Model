import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
import time

# --- Cấu hình ---
MODEL_NAME = 'BAAI/bge-m3'
DB_PATH = './milvus_hannom.db'
COLLECTION_NAME = 'hannom_knowledge'
BATCH_SIZE = 128 # Giảm xuống 128 để tránh OOM nhưng vẫn giữ tải GPU cao (~10-12GB VRAM)

def ingest():
    print(f"🚀 Bắt đầu tiến trình nhúng dữ liệu SIÊU TỐC trên GPU RTX 5060 Ti...")
    
    # 1. Load Model thẳng vào CUDA
    start_time = time.time()
    model = SentenceTransformer(MODEL_NAME, device='cuda')
    print(f"✅ Model {MODEL_NAME} đã tải xong vào GPU trong {time.time() - start_time:.2f}s")

    # 2. Đọc dữ liệu
    data_dir = './data'
    docs = []
    
    # Đọc Thiều Chửu
    tc_path = os.path.join(data_dir, 'Thieu_Chuu_Dictionary.csv')
    if os.path.exists(tc_path):
        df = pd.read_csv(tc_path)
        for _, row in df.iterrows():
            char = str(row.get('char', ''))
            defi = str(row.get('definition', ''))
            if char != 'nan' and defi != 'nan':
                docs.append(f"Chữ: {char}. Nghĩa: {defi}")

    # Đọc CVDICT
    cv_path = os.path.join(data_dir, 'CVDICT_Trung_Viet.csv')
    if os.path.exists(cv_path):
        df = pd.read_csv(cv_path)
        # Giới hạn lấy 50.000 từ đầu tiên để test load GPU cực mạnh
        df = df.head(50000)
        for _, row in df.iterrows():
            trad = str(row.get('Phồn_thể', ''))
            mean = str(row.get('Nghĩa_Việt', ''))
            if trad != 'nan' and mean != 'nan':
                docs.append(f"Từ ghép: {trad}. Nghĩa: {mean}")

    print(f"📊 Đã chuẩn bị {len(docs)} bản ghi để nhúng.")

    # 3. Kết nối Milvus Lite
    client = MilvusClient(DB_PATH)
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    
    # Tạo collection (Milvus Lite tự quản lý schema đơn giản)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=1024 # BGE-M3 có dimension 1024
    )

    # 4. Tiến hành nhúng (Embedding) - Đây là lúc GPU sẽ 'gào thét'
    print(f"🔥 Đang bắt đầu quá trình Embedding {len(docs)} câu trên GPU...")
    
    all_data = []
    for i in range(0, len(docs), 5000): # Chia lô 5000 để insert Milvus
        batch_texts = docs[i : i + 5000]
        
        # Nhúng bằng GPU
        print(f"[*] Đang thực hiện Embedding batch {i//5000 + 1}...")
        embeddings = model.encode(
            batch_texts, 
            batch_size=BATCH_SIZE, 
            show_progress_bar=True,
            device='cuda'
        )
        
        # Format dữ liệu để insert
        for j, emb in enumerate(embeddings):
            all_data.append({
                "id": i + j,
                "vector": emb.tolist(),
                "text": batch_texts[j]
            })

    # 5. Insert vào Milvus theo từng lô nhỏ để tránh "too_many_pings" gRPC
    print(f"📦 Đang đẩy {len(all_data)} vector vào Milvus Lite (chia nhỏ 1000 vector/lần)...")
    for i in range(0, len(all_data), 1000):
        batch = all_data[i : i + 1000]
        client.insert(
            collection_name=COLLECTION_NAME,
            data=batch
        )
        if i % 5000 == 0:
            print(f"✔️ Đã lưu {i + len(batch)} / {len(all_data)} vector...")

    print(f"\n🎉 HOÀN TẤT! Đã nhúng toàn bộ dữ liệu lên GPU trong {time.time() - start_time:.2f}s")
    print(f"Cơ sở dữ liệu lưu tại: {DB_PATH}")

if __name__ == "__main__":
    ingest()
