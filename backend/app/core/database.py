from sqlalchemy import create_all, create_engine
from sqlalchemy.orm import sessionmaker
import os
from app.models.document import Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:admin123@postgres:5432/heritage_db")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    # This will create tables if they don't exist
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
