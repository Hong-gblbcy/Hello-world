from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from typing import List, Optional

from app.models.user import User
from app.core.database import get_db

class UserService:
    """用户服务类"""
    
    @staticmethod
    def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
        """根据ID获取用户"""
        return db.query(User).filter(User.id == user_id).first()
    
    @staticmethod
    def get_user_by_username(db: Session, username: str) -> Optional[User]:
        """根据用户名获取用户"""
        return db.query(User).filter(User.username == username).first()
    
    @staticmethod
    def get_user_by_email(db: Session, email: str) -> Optional[User]:
        """根据邮箱获取用户"""
        return db.query(User).filter(User.email == email).first()
    
    @staticmethod
    def get_all_users(db: Session, skip: int = 0, limit: int = 100) -> List[User]:
        """获取所有用户（分页）"""
        return db.query(User).offset(skip).limit(limit).all()
    
    @staticmethod
    def search_users(db: Session, search_term: str, skip: int = 0, limit: int = 100) -> List[User]:
        """搜索用户（按用户名或邮箱）"""
        return (
            db.query(User)
            .filter(
                (User.username.ilike(f"%{search_term}%")) | 
                (User.email.ilike(f"%{search_term}%"))
            )
            .offset(skip)
            .limit(limit)
            .all()
        )
    
    @staticmethod
    def update_user_role(db: Session, user_id: int, role: str) -> Optional[User]:
        """更新用户角色"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return None
            
            user.role = role
            db.commit()
            db.refresh(user)
            return user
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def delete_user(db: Session, user_id: int) -> bool:
        """删除用户"""
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user:
                return False
            
            db.delete(user)
            db.commit()
            return True
        except SQLAlchemyError as e:
            db.rollback()
            raise e
    
    @staticmethod
    def get_users_by_ids(db: Session, user_ids: List[int]) -> List[User]:
        """根据ID列表获取多个用户"""
        return db.query(User).filter(User.id.in_(user_ids)).all()