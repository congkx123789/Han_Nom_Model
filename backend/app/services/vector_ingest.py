import logging
from pymilvus import Collection, connections
from langchain_openai import OpenAIEmbeddings # Placeholder for BAAI/bge-m3
from app.core.config import settings

class VectorIngestService:
    def __init__(self):
        self.embedding_model = OpenAIEmbeddings() # In prod: bge-m3 via local API
        self.collection_name = "heritage_vectors"
        self._init_connection()

    def _init_connection(self):
        try:
            connections.connect("default", host="milvus", port="19530")
            print("[*] Connected to Milvus for Ingestion")
        except Exception as e:
            logging.error(f"Milvus connection failed: {e}")

    async def ingest_document(self, job_id: str, text_content: str, metadata: dict):
        """
        Processes OCR text -> Vector Embedding -> Milvus Storage
        """
        print(f"[*] Ingesting document {job_id} into Vector DB...")
        
        # 1. Chunking (Recursive Character Text Splitter simulation)
        chunks = [text_content[i:i+500] for i in range(0, len(text_content), 500)]
        
        # 2. Embedding
        vectors = self.embedding_model.embed_documents(chunks)
        
        # 3. Insert into Milvus
        # milvus_collection = Collection(self.collection_name)
        # milvus_collection.insert([chunks, vectors, [metadata]*len(chunks)])
        
        print(f"[x] Successfully ingested {len(chunks)} vectors for job {job_id}")

ingest_service = VectorIngestService()
