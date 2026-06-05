import os
import time
import pandas as pd
from pymilvus import MilvusClient
from langchain_ollama import OllamaEmbeddings

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11435")
MILVUS_URI = os.getenv("MILVUS_URI", "http://milvus_standalone:19530")
COLLECTION_NAME = "heritage_knowledge"

def seed_all():
    print(f"🚀 KÍCH HOẠT NẠP TRI THỨC TOÀN DIỆN (FULL GPU INGESTION)...")
    
    try:
        client = MilvusClient(uri=MILVUS_URI)
        embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_BASE_URL)
        print("✅ Kết nối hệ thống AI GPU thành công.")
    except Exception as e:
        print(f"❌ Khởi động lỗi: {e}")
        return

    # Làm sạch collection để nạp mới toàn bộ
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=768,
        auto_id=True,
        enable_dynamic_field=True
    )

    dictionaries = [
        {"path": "data/Thieu_Chuu_Dictionary.csv", "name": "Từ điển Thiều Chửu"},
        {"path": "data/CVDICT_Trung_Viet.csv", "name": "Từ điển Trung-Việt"},
        {"path": "data/Unihan_Vietnamese.csv", "name": "Dữ liệu Unihan Việt Nam"}
    ]

    total_inserted = 0
    for dict_info in dictionaries:
        path = dict_info['path']
        name = dict_info['name']
        
        if not os.path.exists(path):
            print(f"⚠️ Bỏ qua {name} (Không tìm thấy file).")
            continue

        print(f"📖 Đang xử lý: {name}...")
        try:
            df = pd.read_csv(path)
            data_to_ingest = []
            
            # Tự động hóa việc nhận diện cột để xử lý nhiều loại CSV khác nhau
            cols = df.columns.tolist()
            
            for _, row in df.iterrows():
                # Gom toàn bộ thông tin hàng thành một chuỗi text để AI hiểu bối cảnh
                content_parts = [f"{col}: {row[col]}" for col in cols if pd.notna(row[col])]
                content = " | ".join(content_parts)
                data_to_ingest.append({"text": content, "source": name})

            # Nạp dữ liệu theo từng Batch (100 mục mỗi lần) để không làm quá tải GPU
            batch_size = 100
            for i in range(0, len(data_to_ingest), batch_size):
                batch = data_to_ingest[i:i+batch_size]
                batch_texts = [item['text'] for item in batch]
                
                vectors = embeddings.embed_documents(batch_texts)
                
                insert_data = []
                for j, vector in enumerate(vectors):
                    insert_data.append({
                        "vector": vector,
                        "text": batch[j]['text'],
                        "source": batch[j]['source']
                    })
                
                client.insert(collection_name=COLLECTION_NAME, data=insert_data)
                total_inserted += len(insert_data)
                print(f"   [+] Đã nạp {total_inserted} mục...", end="\r")
            
            print(f"\n✅ Hoàn tất nạp {name}!")

        except Exception as e:
            print(f"❌ Lỗi khi xử lý {name}: {e}")

    print(f"\n🏮 TỔNG CỘNG: Đã nạp {total_inserted} mục tri thức vào GPU memory.")
    print("Hệ thống Hán Nôm của bạn hiện đã đạt trạng thái THÔNG THÁI.")

if __name__ == "__main__":
    seed_all()
