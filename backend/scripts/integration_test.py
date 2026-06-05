import asyncio
import sys
import os

# Thêm đường dẫn vào sys.path để import được app
sys.path.append("/app")

from app.services.rag_engine import rag_engine
from app.services.vector_ingest import ingest_service
from app.db.postgres.session import AsyncSessionLocal
from sqlalchemy import text

async def test_system_integration():
    print("=== BẮT ĐẦU KIỂM TRA HỆ THỐNG (GPU LOCAL MODE) ===")

    # 1. Kiểm tra kết nối AI Model (Ollama GPU)
    print("\n1. Đang kiểm tra AI Model (Chat & Embeddings)...")
    try:
        test_query = "Chữ Hán 'Trần' nghĩa là gì?"
        result = await rag_engine.get_answer(test_query)
        if "answer" in result:
            print(f"✅ AI GPU Phản hồi tốt: {result['answer'][:100]}...")
        else:
            print("❌ AI GPU lỗi phản hồi.")
    except Exception as e:
        print(f"❌ Lỗi kết nối AI Ollama: {e}")

    # 2. Kiểm tra Database (Postgres)
    print("\n2. Đang kiểm tra Database (Postgres)...")
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
            print("✅ Kết nối Postgres ổn định.")
    except Exception as e:
        print(f"❌ Lỗi kết nối Postgres: {e}")

    # 3. Kiểm tra nạp và duyệt dữ liệu (Vector Ingest)
    print("\n3. Đang kiểm tra Vector Database (Milvus Ingest)...")
    try:
        test_text = "Trần Quốc Tuấn là vị tướng vĩ đại của nhà Trần."
        await ingest_service.ingest_document(
            job_id="test_gpu_001",
            text_content=test_text,
            metadata={"source": "test_script"}
        )
        print("✅ Quy trình nạp Vector (Embedding GPU -> Milvus) ổn định.")
    except Exception as e:
        print(f"❌ Lỗi quy trình Ingest: {e}")

    print("\n=== KẾT THÚC KIỂM TRA TOÀN DIỆN ===")

if __name__ == "__main__":
    asyncio.run(test_system_integration())
