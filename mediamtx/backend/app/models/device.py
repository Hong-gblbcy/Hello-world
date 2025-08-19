from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, CheckConstraint
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Device(Base):
    """设备模型"""
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    index_code = Column(String(100), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    ip_address = Column(String(45))
    stream_url = Column(Text)
    status = Column(String(20), default="online", nullable=False)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())

    # 关系定义
    project = relationship("Project", back_populates="devices", lazy="select")

    # 添加检查约束
    __table_args__ = (
        CheckConstraint(
            status.in_(["online", "offline", "maintenance"]),
            name="check_device_status"
        ),
    )

    def __repr__(self):
        return f"<Device(id={self.id}, name={self.name}, status={self.status})>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "index_code": self.index_code,
            "name": self.name,
            "ip_address": self.ip_address,
            "stream_url": self.stream_url,
            "status": self.status,
            "project_id": self.project_id,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

    def is_online(self):
        """检查设备是否在线"""
        return self.status == "online"

    def is_offline(self):
        """检查设备是否离线"""
        return self.status == "offline"

    def is_maintenance(self):
        """检查设备是否在维护中"""
        return self.status == "maintenance"