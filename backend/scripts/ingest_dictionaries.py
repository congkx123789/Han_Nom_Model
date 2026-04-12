import os
import sys
import asyncio
import pandas as pd
from langchain_core.documents import Document

# Thêm đường dẫn backend vào hệ thống để có thể import các module dễ dàng
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from app.services.rag_engine import rag_engine

async def process_and_ingest():
    """ Đọc nội dung CSV, format thành Langchain Document và đẩy vào Milvus """
    if not rag_engine.vector_store:
        print("❌ Lỗi: Milvus chưa được kết nối hoặc cấu hình sai. Vui lòng kiểm tra Docker Milvus và MILVUS_URI.")
        return

    docs = []
    
    # --- 1. Xử lý Thieu_Chuu_Dictionary.csv ---
    thieu_chuu_path = os.path.join(BASE_DIR, '../data/Thieu_Chuu_Dictionary.csv')
    if os.path.exists(thieu_chuu_path):
        print("✅ Tìm thấy bộ từ điển Thiều Chửu. Đang xử lý...")
        df_tc = pd.read_csv(thieu_chuu_path)
        for _, row in df_tc.iterrows():
            char = str(row.get('char', '')).strip()
            pronun = str(row.get('pronunciation', '')).strip()
            defi = str(row.get('definition', '')).strip()
            
            if not char or char == 'nan':
                continue
                
            text_chunk = f"Chữ Hán/Nôm: {char}. Âm Hán Việt: {pronun}. Ý nghĩa theo từ điển Thiều Chửu: {defi}"
            # Gắn siêu dữ liệu (metadata) để sau này lọc riêng nếu cần
            doc = Document(
                page_content=text_chunk, 
                metadata={"source": "Thieu_Chuu_Dictionary", "type": "single_char", "char": char}
            )
            docs.append(doc)
    else:
        print(f"⚠️ Không tìm thấy file tại {thieu_chuu_path}")

    # --- 2. Xử lý CVDICT_Trung_Viet.csv ---
    cvdict_path = os.path.join(BASE_DIR, '../data/CVDICT_Trung_Viet.csv')
    if os.path.exists(cvdict_path):
        print("✅ Tìm thấy bộ đại từ điển CVDICT_Trung_Viet. Đang xử lý... (Sẽ mất thời gian do dung lượng lớn)")
        # CVDICT rất lớn, có thể xử lý Chunk lớn hoặc cắt bớt tùy dung lượng Milvus
        df_cv = pd.read_csv(cvdict_path)
        for _, row in df_cv.iterrows():
            trad = str(row.get('Phồn_thể', '')).strip()
            simp = str(row.get('Giản_thể', '')).strip()
            pinyin = str(row.get('Pinyin', '')).strip()
            meaning = str(row.get('Nghĩa_Việt', '')).strip()
            
            if not trad or trad == 'nan':
                continue
                
            text_chunk = f"Từ vựng tiếng Trung/Hán ghép: {trad} (Giản thể: {simp}). Phiên âm Pinyin: {pinyin}. Nghĩa tiếng Việt theo CVDICT: {meaning}"
            doc = Document(
                page_content=text_chunk, 
                metadata={"source": "CVDICT", "type": "compound_word", "char": trad}
            )
            docs.append(doc)
    else:
        print(f"⚠️ Không tìm thấy file tại {cvdict_path}")

    # --- 3. Bắt đầu quá trình nhúng (Embedding) vào GPU và Insert ---
    total_docs = len(docs)
    if total_docs > 0:
        print(f"\n🚀 Chuẩn bị nhúng {total_docs} documents vào cơ sở dữ liệu Milvus RAG (Vector Store)...")
        print(f"Tiến trình này sẽ chạy embedding model BAAI/bge-m3 trực tiếp trên GPU CUDA của bạn.")
        print("Đã nhận lệnh tăng TỐI ĐA Batch size: Đang chuyển sang Batch = 5000 để ép GPU RTX 5060 Ti chạy hết công suất.")
        
        batch_size = 5000  # Ép GPU chạy mạnh hơn nữa
        for i in range(0, total_docs, batch_size):
            batch = docs[i : i + batch_size]
            print(f"[*] Đang xử lý Embedding cho Batch từ {i} đến {min(i+batch_size, total_docs)}...")
            try:
                rag_engine.vector_store.add_documents(batch)
                print(f"✔️ Đã insert thành công BATCH {i//batch_size + 1} / {(total_docs - 1)//batch_size + 1} ({min(i+batch_size, total_docs)}/{total_docs})")
            except Exception as e:
                print(f"❌ Lỗi khi Insert BATCH {i//batch_size + 1}: {e}")
                
        print("\n🎉 Tiến trình Ingest Data Từ điển đã hoàn tất thành công! Não bộ AI (RAG) đã sẵn sàng hoạt động!")
    else:
        print("\n⚠️ Không có Document nào được trích xuất. Vùi lòng kiểm tra lại đường dẫn file CSV.")

if __name__ == "__main__":
    asyncio.run(process_and_ingest())
