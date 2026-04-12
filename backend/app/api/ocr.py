import shutil
import os
import uuid
from fastapi import APIRouter, File, UploadFile, BackgroundTasks
from app.services import ocr_service
from app.core import storage

router = APIRouter()

@router.post("/process")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Enhanced OCR endpoint: 
    1. Saves file to local temp
    2. Uploads to MinIO
    3. Triggers background Hán-Nôm processing
    """
    job_id = str(uuid.uuid4())
    temp_dir = f"temp/uploads/{job_id}"
    os.makedirs(temp_dir, exist_ok=True)
    
    file_path = os.path.join(temp_dir, file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 1. Store in MinIO
    object_name = f"heritage/scans/{job_id}/{file.filename}"
    storage_url = storage.upload_file(file_path, object_name)
    
    # 2. Trigger background process
    background_tasks.add_task(ocr_service.start_ocr_process, file_path, job_id)

    return {
        "job_id": job_id,
        "filename": file.filename,
        "storage_url": storage_url,
        "status": "processing"
    }
