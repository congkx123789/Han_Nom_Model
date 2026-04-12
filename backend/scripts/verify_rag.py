import asyncio
import sys
import os

# Thêm path backend
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.services.rag_engine import rag_engine

async def verify():
    print("🔍 Đang kiểm tra hệ thống RAG với dữ liệu thật...")
    query = "Chữ nhất nghĩa là gì?"
    try:
        res = await rag_engine.get_answer(query)
        print(f"\n💡 Câu hỏi: {query}")
        print(f"🤖 AI trả lời: {res['answer']}")
        print(f"📚 Số lượng tài liệu tìm thấy: {len(res['context_docs'])}")
        
        if len(res['context_docs']) > 0:
            print("✅ XÁC NHẬN: Hệ thống đã truy xuất dữ liệu từ Milvus thành công!")
        else:
            print("❌ CẢNH BÁO: Milvus không trả về kết quả nào.")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    asyncio.run(verify())
