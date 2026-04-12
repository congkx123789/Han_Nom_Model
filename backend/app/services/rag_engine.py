import logging
import time
import torch
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from pymilvus import MilvusClient
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        # 1. Khởi tạo Embeddings (Direct GPU)
        try:
            print("🚀 Đang khởi động AI Engine: Bge-m3 trên CUDA...")
            self.embed_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
            print("✅ Embeddings GPU đã sẵn sàng.")
        except Exception as e:
            logger.error(f"Embeddings init error: {e}")
            self.embed_model = None

        # 2. Khởi tạo Milvus Client (Lite)
        try:
            self.client = MilvusClient("./milvus_hannom.db")
            self.collection_name = "hannom_knowledge"
            print(f"✅ Kết nối database Milvus: {self.collection_name}")
        except Exception as e:
            logger.error(f"Milvus connect error: {e}")
            self.client = None

        # 3. Khởi tạo Local LLM (Qwen2.5-VL-3B)
        try:
            model_path = "/home/alida/Documents/Cursor/Han_Nom_Model/models/Qwen2.5-VL-3B"
            print(f"🚀 Đang tải Local LLM: Qwen2.5-VL-3B từ {model_path}...")
            
            # Load model với FP16 để tối ưu tốc độ và bộ nhớ
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cuda",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_path)
            print("✅ Local Qwen LLM đã nạp xong vào GPU.")
        except Exception as e:
            logger.error(f"Local LLM init error: {e}")
            self.model = None
            self.processor = None

    async def _classify_intent(self, query: str) -> str:
        """Sử dụng Qwen để phân loại ý định người dùng (Router)"""
        router_prompt = (
            "Phân loại tin nhắn sau đây vào 1 trong 3 nhóm: 'casual', 'dictionary', 'history'.\n"
            "- 'casual': Chào hỏi, cảm ơn, hỏi thăm, hoặc nói chuyện phiếm.\n"
            "- 'dictionary': Tra cứu nghĩa của 1 hoặc vài chữ Hán/Nôm cụ thể.\n"
            "- 'history': Phân tích đoạn văn dài, tìm hiểu bối cảnh lịch sử, dịch văn bản cổ.\n\n"
            "Chỉ trả ra duy nhất 1 từ (casual/dictionary/history).\n\n"
            f"Tin nhắn: {query}"
        )
        
        # Gọi model nhanh để phân loại
        messages = [{"role": "user", "content": [{"type": "text", "text": router_prompt}]}]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        
        intent = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip().lower()
        
        if "dictionary" in intent: return "dictionary"
        if "history" in intent: return "history"
        return "casual"

    async def get_answer(self, query: str) -> Dict[str, Any]:
        if not self.embed_model or not self.client or not self.model:
            return {"answer": "Hệ thống AI chưa sẵn sàng.", "context_docs": []}

        # Lớp 1: Kiểm tra từ khóa nhạy cảm / Vùng cấm (Hard Refusal)
        forbidden_keywords = [
            "hoàng sa", "trường sa", "biển đông", "đường lưỡi bò", "chủ quyền",
            "chính trị", "đảng cộng sản", "tổng bí thư", "chủ tịch nước", "thủ tướng",
            "hồ chí minh", "nguyễn phú trọng", "tô lâm", "phạm minh chính",
            "nhân quyền", "biểu tình", "ngoại giao", "nga-ukraine", "bầu cử",
            "đất đai", "biến đổi khí hậu", "quyền tự do", "tôn giáo", "bitcoin",
            "tiền ảo", "ly hôn", "pháp luật", "thuốc", "bệnh"
        ]
        if any(kw in query.lower() for kw in forbidden_keywords):
            return {
                "answer": "Tôi là trợ lý tra cứu Hán Nôm cổ. Tôi không thể hỗ trợ thảo luận về các chủ đề ngoài phạm vi di sản hoặc các vấn đề chính trị, xã hội, pháp luật hiện đại.",
                "context_docs": []
            }

        try:
            # 0. Kiểm tra Slash Commands
            intent = None
            clean_query = query
            if query.startswith("/dich"):
                intent = "history"
                clean_query = query.replace("/dich", "").strip()
            elif query.startswith("/tracuu"):
                intent = "dictionary"
                clean_query = query.replace("/tracuu", "").strip()
            elif query.startswith("/tomtat"):
                intent = "history"
                clean_query = query.replace("/tomtat", "").strip()

            if not intent:
                intent = await self._classify_intent(query)
            
            # 1. Retrieval & Lớp 3: Score Thresholding
            context_docs = []
            context_text = ""
            
            if intent != "casual":
                query_vector = self.embed_model.encode(query)
                search_res = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector.tolist()],
                    limit=5 if intent == "dictionary" else 8,
                    output_fields=["text"]
                )
                
                # Filter by Threshold (BGE-M3 IP)
                for res in search_res[0]:
                    score = res.get('distance', 0)
                    print(f"🔍 Vector Hit: Score={score:.4f} | Text={res['entity'].get('text', '')[:50]}...")
                    if score > 0.45: # Adjusted from 0.65 for better recall
                        text = res['entity'].get('text', '')
                        context_docs.append({"text": text, "metadata": {"score": score}})
                        context_text += f"- {text}\n"

                # Lớp 3: Nếu không có dữ liệu tin cậy cho Dictionary/History -> Trả về thẳng
                if not context_docs:
                    return {
                        "answer": "Xin lỗi, hiện tại kho dữ liệu Hán Nôm của hệ thống chưa có thông tin về vấn đề này.",
                        "context_docs": []
                    }

            # 2. Xử lý Lớp 1: Strict System Prompt "Vòng Kim Cô"
            common_rules = (
                "QUY TẮC TỐI THƯỢNG:\n"
                "1. Tuyệt đối không bịa đặt sự thật lịch sử (Hallucination). \n"
                "2. Nếu <context> không chứa thông tin cụ thể, bạn PHẢI nói: 'Xin lỗi, hiện tại kho dữ liệu Hán Nôm của hệ thống chưa có thông tin về vấn đề này.'\n"
                "3. ANCHRONISM CHECK: Nếu câu hỏi hỏi về vật thể hiện đại (xe tăng, súng, wifi) gắn với nhân vật cổ, ĐÂY LÀ BẪY. Bắt buộc từ chối và giải thích đây là sự nhầm lẫn thời đại.\n"
                "4. Không thảo luận IT, Y tế, Luật, Nấu ăn, Thể thao.\n"
            )

            if intent == "casual":
                system_prompt = (
                    "Bạn là Trợ lý AI của Nền tảng Di sản Hán Nôm Việt Nam. Nếu người dùng chào hỏi hoặc cảm ơn, "
                    "hãy đáp lại lịch sự, ngắn gọn và nhắc lại chức năng tra cứu Hán Nôm của bạn."
                )
            elif intent == "dictionary":
                system_prompt = (
                    f"Bạn là chuyên gia từ điển Hán Nôm. {common_rules}\n"
                    f"<context>\n{context_text}\n</context>"
                )
            else: # history
                system_prompt = (
                    f"Bạn là học giả nghiên cứu lịch sử và văn bản Hán Nôm. {common_rules}\n"
                    f"<context>\n{context_text}\n</context>"
                )

            # 3. Generation & Lớp 2: Parameter Lock
            messages = [{"role": "user", "content": [{"type": "text", "text": f"{system_prompt}\n\nCâu hỏi: {query}"}]}]
            text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text_prompt], padding=True, return_tensors="pt").to("cuda")

            with torch.no_grad():
                # Ép Temperature = 0.0 và top_p = 0.1
                generated_ids = self.model.generate(
                    **inputs, 
                    max_new_tokens=512,
                    temperature=0.01, # Transformers require temp > 0 for some samplers, or do_sample=False
                    do_sample=False,  # Ngắt hoàn toàn sự ngẫu nhiên
                    top_p=0.1
                )
            
            answer = self.processor.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

            return {"answer": answer, "context_docs": context_docs}

        except Exception as e:
            logger.error(f"RagEngine error: {e}")
            return {"answer": f"Lỗi hệ thống: {str(e)}", "context_docs": []}

rag_engine = RAGEngine()
