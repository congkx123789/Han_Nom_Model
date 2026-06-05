import logging
from typing import List, Dict, Any
from pymilvus import MilvusClient
from langchain_ollama import ChatOllama, OllamaEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(self):
        print("🖥️ Trình xử lý AI: Đang khởi tạo chế độ LOCAL GPU (Ollama)")
        
        # Kết nối tới Ollama GPU trên Host
        ollama_url = settings.OLLAMA_BASE_URL

        # 1. Khởi tạo Embeddings (Ollama GPU)
        try:
            print(f"🚀 Đang khởi động GPU Local Embedding ({ollama_url})...")
            self.embed_model = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=ollama_url
            )
            print("✅ GPU Local Embeddings đã sẵn sàng.")
        except Exception as e:
            logger.error(f"Embeddings init error: {e}")
            self.embed_model = None

        # 2. Khởi tạo Milvus Client
        try:
            self.client = MilvusClient(
                uri=settings.MILVUS_URI,
                token=settings.MILVUS_TOKEN
            )
            self.collection_name = settings.MILVUS_COLLECTION_NAME
            
            # Kiểm tra và tạo collection nếu chưa có
            if not self.client.has_collection(self.collection_name):
                print(f"[*] Tạo collection mới: {self.collection_name} (dim=768)")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    dimension=768, # Nomic dimension
                    auto_id=True,
                    enable_dynamic_field=True
                )
            print(f"✅ Kết nối database Milvus Standalone: {settings.MILVUS_URI}")
        except Exception as e:
            logger.error(f"Milvus connect error: {e}")
            self.client = None

        # 3. Khởi tạo Ollama GPU Chat Model
        try:
            print("🚀 Đang khởi tạo GPU Local Chat Model (1.5B Stable)...")
            self.model = ChatOllama(
                model="qwen2.5:1.5b", # Ultra-stable for restricted environments
                base_url=ollama_url,
                temperature=0.3,
            )
            print("✅ GPU Local AI (Ollama) đã sẵn sàng.")
        except Exception as e:
            logger.error(f"Ollama GPU init error: {e}")
            # Fallback
            self.model = ChatOllama(model="llama3.2:1b", base_url=ollama_url)

    async def _classify_intent(self, query: str) -> str:
        """Sử dụng Gemini để phân loại ý định người dùng (Router)"""
        router_prompt = (
            "Phân loại tin nhắn sau đây vào 1 trong 3 nhóm: 'casual', 'dictionary', 'history'.\n"
            "- 'casual': Chào hỏi, cảm ơn, hỏi thăm, hoặc nói chuyện phiếm.\n"
            "- 'dictionary': Tra cứu nghĩa của 1 hoặc vài chữ Hán/Nôm cụ thể.\n"
            "- 'history': Phân tích đoạn văn dài, tìm hiểu bối cảnh lịch sử, dịch văn bản cổ.\n\n"
            "Chỉ trả ra duy nhất 1 từ (casual/dictionary/history).\n\n"
            f"Tin nhắn: {query}"
        )
        
        try:
            response = await self.model.ainvoke(router_prompt)
            intent = response.content.strip().lower()
            
            if "dictionary" in intent: return "dictionary"
            if "history" in intent: return "history"
            return "casual"
        except Exception as e:
            logger.error(f"Intent classification error: {e}")
            return "casual"

    async def get_answer(self, query: str) -> Dict[str, Any]:
        if not self.embed_model or not self.client or not self.model:
            return {"answer": "Hệ thống AI chưa sẵn sàng (Kiểm tra API Key).", "context_docs": []}

        # Lớp 1: Kiểm tra từ khóa nhạy cảm (Hard Refusal)
        forbidden_keywords = [
            "hoàng sa", "trường sa", "biển đông", "đường lưỡi bò", "chủ quyền",
            "chính trị", "đảng cộng sản", "tổng bí thư", "chủ tịch nước", "thủ tướng",
            "nhân quyền", "biểu tình", "ngoại giao", "nga-ukraine", "bầu cử"
        ]
        if any(kw in query.lower() for kw in forbidden_keywords):
            return {
                "answer": "Tôi là trợ lý tra cứu Hán Nôm cổ. Tôi không thể hỗ trợ thảo luận về các chủ đề ngoài phạm vi di sản hoặc các vấn đề chính trị, xã hội hiện đại.",
                "context_docs": []
            }

        try:
            # 0. Kiểm tra Slash Commands
            intent = None
            if query.startswith("/dich"): intent = "history"
            elif query.startswith("/tracuu"): intent = "dictionary"
            elif query.startswith("/tomtat"): intent = "history"

            if not intent:
                intent = await self._classify_intent(query)
            
            # 1. Retrieval
            context_docs = []
            context_text = ""
            
            if intent != "casual":
                # Thực hiện tìm kiếm Vector
                query_vector = self.embed_model.embed_query(query)
                search_res = self.client.search(
                    collection_name=self.collection_name,
                    data=[query_vector],
                    limit=10 if intent == "dictionary" else 15,
                    output_fields=["text"]
                )
                
                for res in search_res[0]:
                    score = res.get('distance', 0)
                    # Nomic Embeddings: 0.6+ là mức độ liên quan cao (cần chặt chẽ để tránh nhiễu)
                    if score > 0.6: 
                        text = res['entity'].get('text', '')
                        context_docs.append({"text": text, "metadata": {"score": score}})
                        context_text += f"- {text}\n"

                if not context_docs and intent != "casual":
                    logger.warning(f"No high-quality context found for query: {query}")

            # 2. Sinh câu trả lời với System Prompt
            common_rules = (
                "QUY TẮC TỐI THƯỢNG:\n"
                "1. Bạn là chuyên gia Hán Nôm, chỉ được trả lời dựa trên tri thức học thuật chính xác.\n"
                "2. CHỈ TRẢ LỜI DỰA TRÊN DỮ LIỆU TRONG <context>. Nếu <context> không chứa thông tin cụ thể, hãy nói: 'Xin lỗi, hiện tại kho tri thức di sản của tôi chưa có dữ liệu chi tiết về mục này.'\n"
                "3. Tuyệt đối không sử dụng các thuật ngữ hiện đại (như hóa học, vật lý) vào văn bản cổ trừ khi dữ liệu ghi rõ như vậy.\n"
                "4. Không được tự bịa đặt nhân vật hay sự kiện lịch sử.\n"
            )

            if intent == "casual":
                system_prompt = "Bạn là Trợ lý AI Hán Nôm. Hãy chào hỏi và mời người dùng tra cứu tài liệu cổ."
            elif intent == "dictionary":
                system_prompt = f"Bạn là chuyên gia từ điển Hán Nôm. {common_rules}\n<context>\n{context_text}\n</context>"
            else:
                system_prompt = f"Bạn là học giả nghiên cứu lịch sử Hán Nôm. {common_rules}\n<context>\n{context_text}\n</context>"

            response = await self.model.ainvoke(f"{system_prompt}\n\nCâu hỏi: {query}")
            return {"answer": response.content, "context_docs": context_docs}

        except Exception as e:
            logger.error(f"RagEngine error: {e}")
            return {"answer": f"Lỗi hệ thống: {str(e)}", "context_docs": []}

rag_engine = RAGEngine()
