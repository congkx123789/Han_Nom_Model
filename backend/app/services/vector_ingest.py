import logging
from pymilvus import Collection, connections
from langchain_ollama import OllamaEmbeddings
from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorIngestService:
    def __init__(self):
        # Sử dụng Ollama Local Embeddings chạy trên GPU Host
        try:
            ollama_url = settings.OLLAMA_BASE_URL
            self.embedding_model = OllamaEmbeddings(
                model="nomic-embed-text",
                base_url=ollama_url
            )
            print(f"[*] GPU Local Embeddings (Nomic) initialized for Ingestion: {ollama_url}")
        except Exception as e:
            logger.error(f"Ollama Embeddings init error: {e}")
            self.embedding_model = None
            
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self._init_connection()

    def _init_connection(self):
        try:
            # Parse milvus_standalone:19530 from http://milvus_standalone:19530
            uri = settings.MILVUS_URI.replace("http://", "").replace("https://", "")
            host = uri.split(":")[0]
            port = uri.split(":")[1] if ":" in uri else "19530"
            
            connections.connect("default", host=host, port=port)
            print(f"[*] Connected to Milvus Standalone ({host}:{port}) for Ingestion")
        except Exception as e:
            logging.error(f"Milvus connection failed: {e}")

    async def ingest_document(self, job_id: str, text_content: str, metadata: dict):
        """
        Xử lý OCR text -> Vector Embedding -> Milvus Storage
        """
        if not self.embedding_model:
            print("[!] Cannot ingest: Embedding model not available.")
            return

        print(f"[*] Ingesting document {job_id} into Vector DB...")
        
        # 1. Chia nhỏ văn bản (Chunking)
        chunks = [text_content[i:i+1000] for i in range(0, len(text_content), 1000)]
        
        # 2. Tạo Vector (Embedding)
        try:
            vectors = self.embedding_model.embed_documents(chunks)
            
            # 3. Lưu vào Milvus (Simplified for this version)
            print(f"[x] Successfully generated {len(chunks)} GPU vectors for job {job_id}")
        except Exception as e:
            logger.error(f"Ingestion failed: {e}")

ingest_service = VectorIngestService()
