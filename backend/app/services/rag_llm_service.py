from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from app.services.rag_engine import rag_engine

class LlamaRAGService:
    def __init__(self):
        self.llm = ChatOpenAI(model="llama-3", base_url=settings.AI_INFERENCE_URL)
        self.prompt_template = ChatPromptTemplate.from_template("""
        Bạn là một chuyên gia về Di sản Hán-Nôm. Sử dụng ngữ cảnh bên dưới để trả lời câu hỏi của người dùng.
        Nếu bạn không biết, hãy nói rằng bạn không biết.
        
        Ngữ cảnh:
        {context}
        
        Câu hỏi:
        {question}
        
        Câu trả lời (văn phong học thuật):
        """)

    async def answer_question(self, question: str):
        # 1. Retrieve Context from Milvus (Hybrid Search)
        context_docs = await rag_engine.hybrid_search(question)
        context_text = "\n".join([doc["text"] for doc in context_docs])
        
        # 2. Forge Prompt
        prompt = self.prompt_template.format(context=context_text, question=question)
        
        # 3. Call LLM
        # response = await self.llm.invoke(prompt)
        # return response.content
        
        return f"Dựa trên dữ liệu từ {len(context_docs)} nguồn di sản: ... (Kết quả mô phỏng từ Llama-3)"

llama_rag = LlamaRAGService()
