import requests
import json
import time

def test_rag_examples():
    url = "http://localhost:8000/api/v1/chat/"
    
    test_cases = [
        "Nghĩa của chữ 'Quốc' (國) trong từ điển Thiều Chửu?",
        "Giải nghĩa từ ghép 'Sơn Hà' (山河).",
        "Chữ 'Đế' (帝) thường dùng trong ngữ cảnh nào?",
        "Phân biệt chữ 'Nhất' (一) và chữ 'Nhị' (二) về mặt ý nghĩa cơ bản."
    ]
    
    results = []
    
    print(f"🚀 Bắt đầu chạy {len(test_cases)} bài test mẫu...\n")
    
    for i, query in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] Đang truy vấn: {query}")
        start_time = time.time()
        
        try:
            payload = {"query": query, "mode": "bot"}
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                elapsed = time.time() - start_time
                results.append({
                    "query": query,
                    "answer": data.get("text", ""),
                    "latency": f"{elapsed:.2f}s",
                    "sources": [c.get("title") for c in data.get("citations", [])]
                })
                print(f"✅ Xong ({elapsed:.2f}s)")
            else:
                print(f"❌ Lỗi API: {response.status_code}")
                results.append({"query": query, "answer": f"Lỗi API {response.status_code}", "latency": "-", "sources": []})
                
        except Exception as e:
            print(f"❌ Lỗi kết nối: {e}")
            results.append({"query": query, "answer": f"Lỗi: {str(e)}", "latency": "-", "sources": []})
            
    # Lưu kết quả vào file để user xem lại nếu cần
    with open("multi_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    # In ra định dạng Markdown đẹp mắt
    print("\n" + "="*50)
    print("📊 BẢNG KẾT QUẢ KIỂM TRA HÁN-NÔM AI")
    print("="*50 + "\n")
    
    for res in results:
        print(f"### Câu hỏi: {res['query']}")
        print(f"**Độ trễ:** {res['latency']}")
        print(f"**Nguồn trích dẫn:** {', '.join(list(set(res['sources']))[:3])}...")
        print(f"**Trả lời:**\n{res['answer']}\n")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    test_rag_examples()
