from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
from app.core import database, storage

app = FastAPI(
    title="Hán-Nôm Heritage Platform API",
    description="Enterprise-grade API for heritage digitization and document understanding.",
    version="1.0.0"
)

@app.on_event("startup")
def startup_event():
    database.init_db()
    storage.init_storage()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://hannom.lyvuha.com",
        "http://hannom.lyvuha.com",
        "http://localhost:3000",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(router.api_router, prefix="/api/v1")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "heritage-backend"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
