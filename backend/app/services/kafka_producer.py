import datetime
import json
from aiokafka import AIOKafkaProducer
from app.core.config import settings

class KafkaProducer:
    def __init__(self):
        self.producer = None

    async def start(self):
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=settings.KAFKA_BOOTSTRAP_SERVERS
            )
            await self.producer.start()
        except Exception:
            self.producer = None

    async def stop(self):
        if self.producer:
            await self.producer.stop()

    async def send_ocr_job(self, job_id: str, file_url: str):
        if not self.producer:
            return
        message = {
            "job_id": job_id,
            "file_url": file_url,
            "timestamp": str(datetime.datetime.utcnow())
        }
        await self.producer.send_and_wait(
            settings.KAFKA_TOPIC_OCR,
            json.dumps(message).encode("utf-8")
        )

producer = KafkaProducer()
