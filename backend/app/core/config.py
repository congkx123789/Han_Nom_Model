from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hán-Nôm Heritage Platform"
    API_V1_STR: str = "/api/v1"
    
    # PostgreSQL (SQLAlchemy 2.0)
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "admin")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "admin123")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "heritage_db")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "postgres:5432")
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}/{self.POSTGRES_DB}"  # supports host:port in POSTGRES_SERVER

    # Kafka
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    KAFKA_TOPIC_OCR: str = "heritage_ocr_jobs"
    
    # Redis
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    
    # AI Discovery & LLM Configuration
    AI_INFERENCE_URL: str = os.getenv("AI_INFERENCE_URL", "http://localhost:11434/v1")
    
    # Langchain RAG Configuration
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")  # local or openai
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "ollama")  # ollama or openai
    # Model Paths
    YOLO_WEIGHTS_PATH: str = os.getenv("YOLO_WEIGHTS_PATH", "/models/yolo11n.pt")
    QWEN_CHECKPOINT_PATH: str = os.getenv("QWEN_CHECKPOINT_PATH", "/checkpoints/qwen2.5-vl-han-nom")
    FONTS_PATH: str = os.getenv("FONTS_PATH", "/app/assets/fonts/NomNaTong-Regular.ttf")

    MILVUS_URI: str = os.getenv("MILVUS_URI", "http://milvus_standalone:19530")
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "heritage_knowledge")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11435")
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

settings = Settings()
