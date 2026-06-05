import asyncio
import time
import sys
import random

sys.path.append("/app")

from app.services.vector_ingest import ingest_service
from app.services.rag_engine import rag_engine

# Danh sách mẫu để tạo 100 data
sample_texts = [
    "Vua Lê Thái Tổ (黎太祖) tên thật là Lê Lợi, người khởi nghĩa Lam Sơn.",
    "Bản triều Lê (黎朝) là triều đại lâu dài nhất trong lịch sử Việt Nam.",
    "Chữ Nôm (𡨸喃) là hệ chữ tượng hình dùng để viết tiếng Việt.",
    "Nguyễn Trãi (阮廌) là tác giả của Bình Ngô Đại Cáo (平吳大誥).",
    "Sông Bạch Đằng (白藤江) là nơi diễn ra nhiều trận đánh lịch sử.",
    "Kinh đô Thăng Long (昇龍) được lập vào năm 1010 đời Lý Thái Tổ.",
    "Văn Miếu - Quốc Tử Giám (文廟-國子監) là trường đại học đầu tiên.",
    "Truyện Kiều (傳翹) của Nguyễn Du được viết bằng chữ Nôm.",
    "Nhà Trần (陳朝) nổi tiếng với ba lần đánh thắng quân Nguyên Mông.",
    "Hào khí Đông A (東阿) là biểu tượng sức mạnh của triều đại nhà Trần."
]

async def stress_test_100():
    print(f"🔥 BẮT ĐẦU STRESS TEST: 100 LƯỢT DUYỆT DỮ LIỆU BẰNG GPU 🔥")
    start_time = time.time()
    
    success_count = 0
    total_items = 100
    
    for i in range(total_items):
        # Tạo dữ liệu giả lập
        text = f"[{i+1}] {random.choice(sample_texts)}"
        job_id = f"stress_test_{int(time.time())}_{i}"
        
        try:
            # Gửi vào GPU Embedding pipeline
            await ingest_service.ingest_document(
                job_id=job_id,
                text_content=text,
                metadata={"test_id": i, "mode": "stress_test"}
            )
            success_count += 1
            if (i+1) % 10 == 0:
                print(f"🚀 Đã hoàn thành: {i+1}/{total_items}... (RAM/GPU đang tải)")
        except Exception as e:
            print(f"❌ Lỗi tại lượt {i+1}: {e}")

    end_time = time.time()
    duration = end_time - start_time
    
    print("\n" + "="*50)
    print(f"📊 BÁO CÁO HIỆU SUẤT GPU:")
    print(f"- Tổng số mẫu: {total_items}")
    print(f"- Thành công: {success_count}")
    print(f"- Tổng thời gian: {duration:.2f} giây")
    print(f"- Tốc độ trung bình: {duration/total_items:.3f} giây/mẫu")
    print("="*50)

    # Thử tìm kiếm lại 1 mẫu ngẫu nhiên
    print("\n🔍 Đang kiểm tra khả năng truy xuất (RAG)...")
    search_res = await rag_engine.get_answer("Ai là người khởi nghĩa Lam Sơn?")
    print(f"✅ Kết quả truy xuất: {search_res['answer'][:150]}...")

if __name__ == "__main__":
    asyncio.run(stress_test_100())
