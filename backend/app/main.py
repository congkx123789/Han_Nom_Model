import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.db.postgres.session import init_db
from app.api.v1.endpoints import ocr, chat
from app.api.v1.websockets import progress
from app.services.kafka_producer import producer

logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict to hannom.lyvuha.com
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await init_db()
    try:
        await producer.start()
    except Exception as exc:
        logger.warning("Kafka producer unavailable at startup: %s", exc)
    # Note: Kafka Consumer (Worker) should run in a separate process/container in production

@app.on_event("shutdown")
async def shutdown_event():
    await producer.stop()

# Routers
app.include_router(ocr.router, prefix=f"{settings.API_V1_STR}/ocr", tags=["ocr"])
app.include_router(chat.router, prefix=f"{settings.API_V1_STR}/chat", tags=["chat"])
app.include_router(progress.router, prefix="/ws/progress", tags=["websockets"])

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "4.0.0-SOA"}
