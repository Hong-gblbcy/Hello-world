from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from app.core.database import Base

class User(Base):
    """用户模型"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    role = Column(String(20), nullable=False, default="user")
    created_at = Column(DateTime, default=func.now())

    def __repr__(self):
        return f"<User(id={self.id}, username={self.username}, role={self.role})>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def is_superuser(self):
        """检查是否为超级用户"""
        return self.role == "superuser"

    def is_admin(self):
        """检查是否为管理员"""
        return self.role in ["superuser", "admin"]