import uuid
from fastapi import APIRouter, UploadFile, File, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.postgres.session import get_db
from app.db.postgres.models import Document, JobStatus
from app.services.kafka_producer import producer
from app.core import storage

router = APIRouter()

@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Enterprise Upload:
    1. Save to MinIO
    2. Create DB record (PENDING)
    3. Push event to Kafka
    """
    job_id = str(uuid.uuid4())
    
    # 1. Upload to MinIO
    object_name = f"heritage/scans/{job_id}/{file.filename}"
    # In a real enterprise setup, we'd use a presigned URL as requested
    storage_url = await storage.upload_file_async(file, object_name)
    
    # 2. Save Metadata to Postgres (SQLAlchemy 2.0)
    new_doc = Document(
        job_id=job_id,
        filename=file.filename,
        storage_url=storage_url,
        status=JobStatus.PENDING
    )
    db.add(new_doc)
    await db.commit()
    
    # 3. Push to Kafka for Async OCR Processing
    await producer.send_ocr_job(job_id, storage_url)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "storage_url": storage_url
    }
