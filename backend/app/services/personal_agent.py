from typing import Annotated
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import pipeline
import torch

from app.db.personal_db import get_user_profile_json

# ------------- LỚP 2: CÔNG CỤ AI (LANGCHAIN TOOLS) -------------

@tool("fetch_personal_context")
def fetch_personal_context(user_id: str) -> str:
    """
    Công cụ để lấy thông tin cá nhân/hồ sơ nghiên cứu của người dùng từ hệ thống cơ sở dữ liệu.
    CHỈ GỌI công cụ này nếu câu hỏi của người dùng mang tính cá nhân, yêu cầu tư vấn chuyên ngành,
    hoặc hỏi 'Theo chuyên môn của tôi thì sao?', 'Tôi là ai'.
    """
    return get_user_profile_json(user_id)

tools = [fetch_personal_context]

# ------------- LỚP 3: LUỒNG ĐỊNH TUYẾN Ý ĐỊNH & TOKEN SAVING -------------

# System Prompt kết hợp (Vòng Kim Cô + Agentic Tool Calling)
DYNAMIC_SYSTEM_PROMPT = """Bạn là trợ lý hệ thống Hán Nôm học cao cấp chạy trên nhân Qwen 2.5.
Bạn có quyền truy cập công cụ `fetch_personal_context`. 

QUY TẮC SỬ DỤNG TOOL KIỂM SOÁT TOKEN:
1. Nếu câu hỏi của người dùng mang tính cá nhân (VD: "Theo chuyên môn của tôi thì sao?", "Nhắc lại mục tiêu nghiên cứu của tôi", "Tôi là ai"), HÃY GỌI CÔNG CỤ `fetch_personal_context` để lấy dữ liệu.
2. Nếu câu hỏi CHỈ LÀ tra cứu Hán Nôm thông thường (VD: "Dịch chữ Nhẫn"), KHÔNG ĐƯỢC GỌI CÔNG CỤ để tiết kiệm tài nguyên. Trả lời ngay bằng kiến thức hiện có.

Khi có dữ liệu từ Tool, hãy điều chỉnh LỜI LẼ, GIỌNG ĐIỆU và ĐỘ SÂU của câu trả lời sao cho khớp hoàn toàn với `research_focus` và `preferred_explanation_level` của người dùng.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", DYNAMIC_SYSTEM_PROMPT),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ------------- LỚP 4: XỬ LÝ KẾT QUẢ (CHAIN EXECUTION) -------------

class PersonalAgentManager:
    def __init__(self, llm_model_path: str):
        print(f"🚀 Nạp Mô hình Qwen cho LangChain Agent từ {llm_model_path}")
        # Khởi tạo Local Pipeline (Vì Qwen 2.5 hỗ trợ Function Calling bản địa cực tốt)
        # Lưu ý: Cần wrapping model Qwen2.5 theo Interface ChatModel của LangChain.
        # Ở đây minh hoạ dùng HuggingFacePipeline (Trong thực tế có thể dùng ChatHuggingFace để hỗ trợ tính năng bind_tools)
        
        # NOTE: Giả lập khởi tạo LLM. Trên thực tế bạn sẽ truyền self.model từ RAGEngine vào.
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
        from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
        
        # Đoạn code chuẩn của LangChain để chạy Tool Calling với Qwen Local:
        # llm = ChatHuggingFace(pipeline=pipeline("text-generation", model=llm_model_path, model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"}))
        
        # Để code không bị lỗi compile trong sandbox, tạo Placeholder Mock:
        class MockQwenWithTools:
            def bind_tools(self, tools):
                return self
        
        # llm = MockQwenWithTools()  <-- Un-comment dòng dưới khi ráp vào hệ thống thật
        llm = MockQwenWithTools() 
        
        # Tạo Agent
        self.agent = create_tool_calling_agent(llm, tools, prompt)
        
        # AgentExecutor: Quản lý luồng (Xác định xem có cần Call Tool không, ném vào ScratchPad, Loop lại)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=tools, 
            verbose=True, # Bật mode này để log ra Terminal coi Qwen gọi Tool xịn cỡ nào
            handle_parsing_errors=True
        )

    def ask(self, query: str, user_id: str = "user_123") -> str:
        """
        Hàm chính chạy luồng Agent.
        - Bước 1: Qwen đọc query, quyết định gọi tool hay không.
        - Bước 2: Gọi Tool `fetch_personal_context("user_123")` kéo về Json (Nếu cần).
        - Bước 3: Generate câu hỏi cuối cùng dựa theo profile vừa kéo.
        """
        # Inject mã người dùng vào input để Tool biết đường gọi DB
        input_with_context = f"[Context: User đang giao tiếp có mã ID là '{user_id}']\nCâu hỏi: {query}"
        
        print(f"\n--- NHẬN CÂU HỎI TỪ {user_id} ---")
        response = self.agent_executor.invoke({"input": input_with_context})
        return response["output"]

# ==========================================
# CÁCH SỬ DỤNG TRONG FASTAPI
# 
# agent_manager = PersonalAgentManager("/models/Qwen2.5-3B")
# 
# @app.post("/api/v1/chat/")
# async def chat(query: str, current_user = Depends(get_current_user)):
#     answer = agent_manager.ask(query, user_id=current_user.id)
#     return {"text": answer}
# ==========================================
