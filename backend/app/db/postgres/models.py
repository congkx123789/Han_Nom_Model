from sqlalchemy import Column, Integer, String, DateTime, JSON, Enum
from app.db.postgres.session import Base
import enum
import datetime

class JobStatus(enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String, unique=True, index=True)
    filename = Column(String)
    storage_url = Column(String)
    status = Column(Enum(JobStatus), default=JobStatus.PENDING)
    metadata_json = Column(JSON, nullable=True)
    ocr_result = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.datetime.utcnow)
