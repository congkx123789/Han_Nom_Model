from typing import Annotated
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from app.core.config import settings
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

# Qwen 2.5 14B hỗ trợ tool calling rất tốt
DYNAMIC_SYSTEM_PROMPT = """Bạn là trợ lý hệ thống Hán Nôm học cao cấp chạy trên nhân Local GPU (Qwen 2.5).
Bạn có quyền truy cập công cụ `fetch_personal_context`. 

QUY TẮC SỬ DỤNG TOOL:
1. Nếu câu hỏi mang tính cá nhân hoặc hỏi về hồ sơ người dùng, hãy dùng `fetch_personal_context`.
2. Nếu câu hỏi là tra cứu thông thường, trả lời ngay bằng kiến thức hiện có.

Hãy điều chỉnh lời lẽ và độ sâu chuyên môn khớp với research_focus của người dùng.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", DYNAMIC_SYSTEM_PROMPT),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# ------------- LỚP 4: XỬ LÝ KẾT QUẢ (CHAIN EXECUTION) -------------

class PersonalAgentManager:
    def __init__(self):
        print("🚀 Khởi tạo Local GPU Personal Agent (Ollama)...")
        
        # Kết nối tới Ollama GPU trên Host
        ollama_url = "http://host.docker.internal:11435"
        
        self.llm = ChatOllama(
            model="qwen2.5:1.5b",
            base_url=ollama_url,
            temperature=0
        )
        
        # Tạo Agent
        self.agent = create_tool_calling_agent(self.llm, tools, prompt)
        
        # AgentExecutor: Quản lý luồng
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=tools, 
            verbose=True,
            handle_parsing_errors=True
        )

    def ask(self, query: str, user_id: str = "user_123") -> str:
        """
        Hàm chính chạy luồng Agent.
        """
        input_with_context = f"[Context: User ID: '{user_id}']\nCâu hỏi: {query}"
        
        print(f"\n--- NHẬN CÂU HỎI TỪ {user_id} ---")
        response = self.agent_executor.invoke({"input": input_with_context})
        return response["output"]

# Khởi tạo singleton manager
try:
    personal_agent_manager = PersonalAgentManager()
except Exception as e:
    print(f"Failed to init PersonalAgentManager: {e}")
    personal_agent_manager = None
