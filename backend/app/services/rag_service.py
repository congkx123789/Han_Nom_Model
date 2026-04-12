from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.core import config
import os

def get_rag_response(query: str, context: str):
    """
    RAG lookup using Llama-3 (local) or OpenAI placeholder.
    As described in promt_process_made_web.txt (Llama-3 8B).
    """
    prompt = ChatPromptTemplate.from_template("""
    You are an expert scholar in Hán-Nôm linguistic history.
    Based on the following historical context, translate and explain the query.
    
    Context: {context}
    Query: {query}
    
    Provide the Hán-Việt reading, the modern Vietnamese meaning, and any relevant historical notes.
    """)
    
    # Placeholder for local Llama-3 via Ollama or vLLM
    model = ChatOpenAI(
        base_url="http://localhost:11434/v1", # Default Ollama port
        api_key="ollama",
        model="llama3"
    )
    
    chain = prompt | model | StrOutputParser()
    
    return chain.invoke({"query": query, "context": context})

def search_vector_db(query: str):
    """
    Placeholder for Milvus search using BAAI/bge-m3 embeddings.
    """
    # Logic to embed query and search Milvus
    return "Sample history context from Milvus..."
