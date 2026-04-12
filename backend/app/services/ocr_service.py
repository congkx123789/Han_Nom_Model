import subprocess
import os
from app.core import storage

def start_ocr_process(file_path: str, job_id: str):
    """
    Triggers the existing YOLO + Qwen2.5-VL pipeline.
    This runs synchronously in the background task worker.
    """
    # 1. Store in MinIO
    object_name = f"jobs/{job_id}/{os.path.basename(file_path)}"
    storage_url = storage.upload_file(file_path, object_name)
    
    # 2. Run existing script (example path from our reorganization)
    # Using the script we reorganized earlier: scripts/inference/yolo_qwen_pipeline.py
    script_path = "/home/alida/Documents/Cursor/Han_Nom_Model/scripts/inference/yolo_qwen_pipeline.py"
    
    command = [
        "python3", script_path,
        "--input", file_path,
        "--output", f"output/jobs/{job_id}",
        "--weights", "/home/alida/Documents/Cursor/Han_Nom_Model/models/yolo11n.pt"
    ]
    
    try:
        # We run this as a subprocess to keep the main API alive
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return {"status": "success", "output": result.stdout, "storage_url": storage_url}
    except subprocess.CalledProcessError as e:
        return {"status": "error", "error": e.stderr}
