import asyncio
import sys
import os

# Thêm path backend để import
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from app.services.rag_engine import rag_engine

async def verify():
    print("====================================")
    print("🔍 HÁN-NÔM AI BENCHMARK SUITE")
    print("====================================")
    print("\n[Bước 1] Khởi Động Engine AI...")
    
    if not rag_engine.model or not rag_engine.embed_model:
        print("❌ LỖI: Không thể kết nối với VRAM Model.")
        return

    queries = [
        "Chữ 'nhất' nghĩa là gì?",
    ]

    for i, q in enumerate(queries):
        print(f"\n[{i+1}/2] TEST CASE: {q}")
        print("-" * 40)
        try:
            start = asyncio.get_event_loop().time()
            res = await rag_engine.get_answer(q)
            latency = asyncio.get_event_loop().time() - start
            
            print(f"⏱️ RAG Latency: {latency:.2f}s")
            print(f"📚 Context Docs: {len(res['context_docs'])} (Từ Milvus)")
            print(f"🤖 Qwen2.5 Answer:\n\n{res['answer']}\n")
        except Exception as e:
            print(f"❌ LỖI TRONG QUÁ TRÌNH TRUY VẤN: {e}")

    print("====================================")
    print("✅ HOÀN TẤT BÀI KIỂM TRA CHO LLM")
    
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    asyncio.run(verify())
