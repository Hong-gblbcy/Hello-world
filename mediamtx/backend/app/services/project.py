from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional

from app.models.project import Project, ProjectUser
from app.models.user import User
from app.schemas.project import ProjectCreate, ProjectUpdate, ProjectUserCreate
from app.core.database import get_db

class ProjectService:
    """项目服务类"""
    
    @staticmethod
    def create_project(db: Session, project_data: ProjectCreate, created_by: int) -> Project:
        """创建新项目"""
        try:
            project = Project(
                name=project_data.name,
                description=project_data.description,
                created_by=created_by
            )
            db.add(project)
            db.commit()
            db.refresh(project)
            
            # 自动将创建者添加为项目管理员
            ProjectService.add_project_member(db, project.id, created_by, "admin")
            
            return project
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def get_project(db: Session, project_id: int) -> Optional[Project]:
        """根据ID获取项目"""
        return db.query(Project).filter(Project.id == project_id).first()
    
    @staticmethod
    def get_user_projects(db: Session, user_id: int) -> List[Project]:
        """获取用户参与的所有项目"""
        return (
            db.query(Project)
            .join(ProjectUser, ProjectUser.project_id == Project.id)
            .filter(ProjectUser.user_id == user_id)
            .all()
        )
    
    @staticmethod
    def update_project(db: Session, project_id: int, project_data: ProjectUpdate) -> Optional[Project]:
        """更新项目信息"""
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                return None
            
            if project_data.name is not None:
                project.name = project_data.name
            if project_data.description is not None:
                project.description = project_data.description
            
            db.commit()
            db.refresh(project)
            return project
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def delete_project(db: Session, project_id: int) -> None:
        """删除项目"""
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                raise ValueError("项目不存在")
            
            db.delete(project)
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def add_project_member(db: Session, project_id: int, user_id: int, role: str = "member") -> Optional[ProjectUser]:
        """添加项目成员"""
        try:
            # 检查用户是否已经是项目成员
            existing_member = (
                db.query(ProjectUser)
                .filter(ProjectUser.project_id == project_id, ProjectUser.user_id == user_id)
                .first()
            )
            if existing_member:
                return existing_member
            
            project_member = ProjectUser(
                project_id=project_id,
                user_id=user_id,
                role=role
            )
            db.add(project_member)
            db.commit()
            db.refresh(project_member)
            return project_member
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def remove_project_member(db: Session, project_id: int, user_id: int) -> None:
        """移除项目成员"""
        try:
            project_member = (
                db.query(ProjectUser)
                .filter(ProjectUser.project_id == project_id, ProjectUser.user_id == user_id)
                .first()
            )
            if not project_member:
                raise ValueError("项目成员不存在")
            
            db.delete(project_member)
            db.commit()
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def get_project_members(db: Session, project_id: int) -> List[ProjectUser]:
        """获取项目所有成员"""
        return (
            db.query(ProjectUser)
            .filter(ProjectUser.project_id == project_id)
            .all()
        )
    
    @staticmethod
    def is_project_member(db: Session, project_id: int, user_id: int) -> bool:
        """检查用户是否是项目成员"""
        return (
            db.query(ProjectUser)
            .filter(ProjectUser.project_id == project_id, ProjectUser.user_id == user_id)
            .first() is not None
        )
    
    @staticmethod
    def is_project_admin(db: Session, project_id: int, user_id: int) -> bool:
        """检查用户是否是项目管理员"""
        member = (
            db.query(ProjectUser)
            .filter(ProjectUser.project_id == project_id, ProjectUser.user_id == user_id)
            .first()
        )
        return member is not None and member.role == "admin"
    
    @staticmethod
    def get_user_role_in_project(db: Session, project_id: int, user_id: int) -> Optional[str]:
        """获取用户在项目中的角色"""
        member = (
            db.query(ProjectUser)
            .filter(ProjectUser.project_id == project_id, ProjectUser.user_id == user_id)
            .first()
        )
        return member.role if member else None