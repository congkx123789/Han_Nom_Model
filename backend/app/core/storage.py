from minio import Minio
from app.core import config
import os

client = Minio(
    "localhost:9000",
    access_key=os.getenv("MINIO_ROOT_USER", "minioadmin"),
    secret_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin"),
    secure=False
)

BUCKET_NAME = "heritage-documents"

def init_storage():
    if not client.bucket_exists(BUCKET_NAME):
        client.make_bucket(BUCKET_NAME)
        print(f"Created bucket: {BUCKET_NAME}")

def upload_file(file_path: str, object_name: str):
    client.fput_object(BUCKET_NAME, object_name, file_path)
    return f"http://localhost:9000/{BUCKET_NAME}/{object_name}"

def upload_stream(file_stream, size, object_name: str):
    client.put_object(BUCKET_NAME, object_name, file_stream, size)
    return f"http://localhost:9000/{BUCKET_NAME}/{object_name}"
