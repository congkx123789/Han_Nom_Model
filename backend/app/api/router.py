from fastapi import APIRouter
from app.api import ocr, dictionary

api_router = APIRouter()

api_router.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
api_router.include_router(dictionary.router, prefix="/dictionary", tags=["Dictionary"])
