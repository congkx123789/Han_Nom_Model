import json
import asyncio
from aiokafka import AIOKafkaConsumer
from app.core.config import settings
from app.services.ocr_service import start_ocr_process
from app.services.vector_ingest import ingest_service

async def ocr_worker():
    consumer = AIOKafkaConsumer(
        settings.KAFKA_TOPIC_OCR,
        bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS,
        group_id="heritage_ocr_workers"
    )
    await consumer.start()
    try:
        async for msg in consumer:
            job_data = json.loads(msg.value.decode("utf-8"))
            job_id = job_data["job_id"]
            file_url = job_data["file_url"]
            
            print(f"[*] Processing OCR Job: {job_id}")
            # 1. Trigger ML Pipeline (YOLO + Qwen)
            # This is a sync subprocess, run in thread to avoid blocking loop
            result = await asyncio.to_thread(start_ocr_process, file_url, job_id)
            
            # 2. Ingest into Vector DB for RAG
            await ingest_service.ingest_document(
                job_id=job_id, 
                text_content=result.get("text", ""),
                metadata={"filename": job_data.get("filename"), "era": "Lý"} # Era detection simulation
            )

            # 3. Update Database via Async Session
            
    finally:
        await consumer.stop()

if __name__ == "__main__":
    print("📢 AI Worker is starting... Listening for Kafka messages.")
    asyncio.run(ocr_worker())
