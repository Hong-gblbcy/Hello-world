from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base

class Project(Base):
    """项目模型"""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text)
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime, default=func.now())

    # 关系定义
    creator = relationship("User", backref="created_projects")
    devices = relationship("Device", back_populates="project", cascade="all, delete-orphan", lazy="select")
    members = relationship("ProjectUser", back_populates="project", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Project(id={self.id}, name={self.name})>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }

class ProjectUser(Base):
    """项目用户关联模型"""
    __tablename__ = "project_users"

    id = Column(Integer, primary_key=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    role = Column(String(20), nullable=False, default="member")
    created_at = Column(DateTime, default=func.now())

    # 关系定义
    project = relationship("Project", back_populates="members")
    user = relationship("User", backref="project_memberships")

    def __repr__(self):
        return f"<ProjectUser(project_id={self.project_id}, user_id={self.user_id}, role={self.role})>"

    def to_dict(self):
        """转换为字典格式"""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "user_id": self.user_id,
            "role": self.role,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }