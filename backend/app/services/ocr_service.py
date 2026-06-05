import os
import requests
import base64
from app.core import storage
from app.core.config import settings

def start_ocr_process(file_path: str, job_id: str):
    """
    Xử lý OCR hoàn toàn LOCAL GPU sử dụng Ollama (Llama 3.2 Vision).
    Thay thế Gemini Cloud bằng sức mạnh phần cứng tại chỗ.
    """
    # 1. Store in MinIO
    object_name = f"jobs/{job_id}/{os.path.basename(file_path)}"
    storage_url = storage.upload_file(file_path, object_name)
    
    try:
        # Load image and encode to Base64
        with open(file_path, "rb") as f:
            image_bytes = f.read()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
        print(f"[*] Gửi ảnh {file_path} tới Ollama (GPU) để bóc tách chữ...")
        
        ollama_url = f"{settings.OLLAMA_BASE_URL}/api/generate"
        prompt = """Bạn là một chuyên gia OCR Hán Nôm cao cấp. Hãy trích xuất toàn bộ văn bản Hán Nôm có trong ảnh này.
        - Đọc theo thứ tự từ trên xuống dưới, từ phải sang trái (nếu là dạng cột).
        - Chỉ trả về nội dung văn bản bóc tách được, không thêm giải thích hay mô tả.
        - Nếu có chữ không rõ, dùng ký tự '□'.
        """
        
        payload = {
            "model": "llama3.2-vision",
            "prompt": prompt,
            "images": [image_base64],
            "stream": False
        }
        
        response = requests.post(ollama_url, json=payload, timeout=180)
        response.raise_for_status()
        
        ocr_text = response.json().get("response", "").strip()
        print(f"[x] Local GPU OCR Success: {ocr_text[:50]}...")
        
        return {
            "status": "success", 
            "text": ocr_text, 
            "storage_url": storage_url,
            "job_id": job_id
        }
        
    except Exception as e:
        print(f"[!] Local GPU OCR Error: {e}")
        return {"status": "error", "error": str(e)}
