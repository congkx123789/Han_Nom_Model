from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from app.services.rag_engine import rag_engine

router = APIRouter()

class Citation(BaseModel):
    title: str
    href: str

class ChatMessage(BaseModel):
    role: str
    text: str
    citations: Optional[List[Citation]] = []

class ChatRequest(BaseModel):
    query: str
    mode: str = "bot"

class ContextBlock(BaseModel):
    title: str
    content: str | list

class ChatResponse(BaseModel):
    text: str
    citations: List[Citation] = []
    contextual_data: Optional[dict] = None

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Receives a message from the frontend and uses the RAG Engine (LangChain)
    to generate an accurate response and real citations from Milvus.
    """
    try:
        if request.mode == "db":
            # Just do a raw semantic search and return the top docs
            docs = await rag_engine.hybrid_search(request.query)
            text_response = f"Kết quả từ Vector RAG Search: Tìm thấy {len(docs)} tài liệu."
            
            citations = []
            for doc in docs:
                source = doc.get("metadata", {}).get("source", "Delta Lake/Milvus")
                citations.append(Citation(title=source, href="#"))
                
            return ChatResponse(
                text=text_response,
                citations=citations,
                contextual_data={
                    "char_lookup": {"char": "N/A", "hanviet": "-", "meaning": "Kết quả DB Search thô"},
                    "structure": {"radicals": "-", "variants": "-"},
                    "examples": [d.get("text", "")[:100] + "..." for d in docs]
                }
            )
            
        else:
            # Query the Local LLM / Cloud LLM RAG Pipeline
            result = await rag_engine.get_answer(request.query)
            
            citations = []
            for doc in result.get("context_docs", []):
                source = doc.get("metadata", {}).get("source", "Milvus Context")
                citations.append(Citation(title=source, href="#"))
            
            if not citations:
                citations.append(Citation(title="AI Generated", href="#"))
                
            # Build Context Data Panel dynamically based on results
            contextual_data = {
                "char_lookup": {
                    "char": "Đang tra" if result.get("context_docs") else "N/A",
                    "hanviet": "Chi tiết RAG bên dưới",
                    "meaning": "Tìm kiếm ngữ nghĩa hoàn tất."
                },
                "structure": {
                    "radicals": "Tra cứu động",
                    "variants": "Chi tiết LLM"
                },
                "examples": [
                    d.get("text", "...")[:80] + "..." 
                    for d in result.get("context_docs", [])[:3]
                ]
            }
            
            return ChatResponse(
                text=result.get("answer", "Xin lỗi, AI tạm thời mất kết nối."),
                citations=citations,
                contextual_data=contextual_data
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
