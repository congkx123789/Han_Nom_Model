import json
from sqlalchemy import Column, String, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# ------------- LỚP 1: DATABASE MANAGEMENT -------------
# (Service-Oriented Architecture: Tách riêng cá nhân hóa thành DB độc lập)

# Cấu hình PostgreSQL (Có thể cấu hình qua .env)
# Ví dụ: postgresql://user:password@localhost/hannom_db
DATABASE_URL = "sqlite:///./personal_profiles.db" # Dùng tạm SQLite để minh họa, thay bằng chuỗi PGSQL thật khi deploy

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserProfile(Base):
    """
    Bảng lưu trữ hồ sơ cá nhân của người dùng.
    Dùng để cung cấp ngữ cảnh (Dynamic Profiling) cho Qwen.
    """
    __tablename__ = "user_profiles"

    user_id = Column(String, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    academic_background = Column(String, nullable=True) # Học vấn / Trường học
    research_focus = Column(String, nullable=True)      # Hướng nghiên cứu trọng tâm
    preferred_explanation_level = Column(String, nullable=True) # Mức độ giải thích (Học thuật hay Đơn giản)

# Tạo bảng (init db)
Base.metadata.create_all(bind=engine)

def get_user_profile_json(user_id: str) -> str:
    """Hàm Helper: Kéo dữ liệu từ DB và trả về JSON."""
    db = SessionLocal()
    try:
        user = db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if user:
            data = {
                "user_id": user.user_id,
                "full_name": user.full_name,
                "academic_background": user.academic_background,
                "research_focus": user.research_focus,
                "preferred_explanation_level": user.preferred_explanation_level
            }
            return json.dumps(data, ensure_ascii=False)
        return json.dumps({"error": "Không tìm thấy hồ sơ cá nhân cho user này."})
    except Exception as e:
        return json.dumps({"error": str(e)})
    finally:
        db.close()

# Khởi tạo một User mặc định để test nếu Database trống
def seed_test_user():
    db = SessionLocal()
    if not db.query(UserProfile).first():
        test_user = UserProfile(
            user_id="user_123",
            full_name="Nguyễn Văn A",
            academic_background="Nghiên cứu sinh Viện Hán Nôm",
            research_focus="Huấn luyện mô hình AI bóc tách OCR chữ Nôm",
            preferred_explanation_level="Học thuật chuyên sâu, tập trung vào kỹ thuật và cấu trúc chữ."
        )
        db.add(test_user)
        db.commit()
    db.close()

seed_test_user()
