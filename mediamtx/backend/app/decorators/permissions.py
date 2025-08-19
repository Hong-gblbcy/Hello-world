from functools import wraps
from fastapi import HTTPException, status, Depends
from typing import Callable, Any, Optional

from app.services.auth import get_current_user
from app.models.user import User

def require_role(required_role: str):
    """
    要求特定角色的装饰器
    
    Args:
        required_role: 需要的角色 (superuser, admin, user)
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            current_user = kwargs.get('current_user')
            if not current_user:
                # 如果没有current_user参数，尝试从依赖获取
                try:
                    current_user = await get_current_user()
                except:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="需要认证"
                    )
            
            if required_role == "superuser" and not current_user.is_superuser():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="需要超级用户权限"
                )
            
            if required_role == "admin" and not current_user.is_admin():
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="需要管理员权限"
                )
            
            if required_role == "user" and current_user.role not in ["user", "admin", "superuser"]:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="需要用户权限"
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_superuser(func: Callable) -> Callable:
    """要求超级用户权限的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        current_user = kwargs.get('current_user')
        if not current_user:
            try:
                current_user = await get_current_user()
            except:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要认证"
                )
        
        if not current_user.is_superuser():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要超级用户权限"
            )
        
        return await func(*args, **kwargs)
    return wrapper

def require_admin(func: Callable) -> Callable:
    """要求管理员权限的装饰器"""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        current_user = kwargs.get('current_user')
        if not current_user:
            try:
                current_user = await get_current_user()
            except:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="需要认证"
                )
        
        if not current_user.is_admin():
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="需要管理员权限"
            )
        
        return await func(*args, **kwargs)
    return wrapper

def require_project_member(project_id_param: str = "project_id"):
    """
    要求项目成员权限的装饰器
    
    Args:
        project_id_param: 项目ID参数名
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            from app.core.database import get_db
            from app.services.project import ProjectService
            
            current_user = kwargs.get('current_user')
            if not current_user:
                try:
                    current_user = await get_current_user()
                except:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="需要认证"
                    )
            
            # 获取项目ID
            project_id = kwargs.get(project_id_param)
            if project_id is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="缺少项目ID参数"
                )
            
            # 检查用户是否是项目成员
            db = kwargs.get('db')
            if not db:
                # 如果没有db参数，尝试从依赖获取
                from app.core.database import SessionLocal
                db = SessionLocal()
                try:
                    if not ProjectService.is_project_member(db, project_id, current_user.id):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="需要项目成员权限"
                        )
                finally:
                    db.close()
            else:
                if not ProjectService.is_project_member(db, project_id, current_user.id):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="需要项目成员权限"
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def require_project_admin(project_id_param: str = "project_id"):
    """
    要求项目管理员权限的装饰器
    
    Args:
        project_id_param: 项目ID参数名
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            from app.core.database import get_db
            from app.services.project import ProjectService
            
            current_user = kwargs.get('current_user')
            if not current_user:
                try:
                    current_user = await get_current_user()
                except:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="需要认证"
                    )
            
            # 获取项目ID
            project_id = kwargs.get(project_id_param)
            if project_id is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="缺少项目ID参数"
                )
            
            # 检查用户是否是项目管理员
            db = kwargs.get('db')
            if not db:
                # 如果没有db参数，尝试从依赖获取
                from app.core.database import SessionLocal
                db = SessionLocal()
                try:
                    if not ProjectService.is_project_admin(db, project_id, current_user.id):
                        raise HTTPException(
                            status_code=status.HTTP_403_FORBIDDEN,
                            detail="需要项目管理员权限"
                        )
                finally:
                    db.close()
            else:
                if not ProjectService.is_project_admin(db, project_id, current_user.id):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="需要项目管理员权限"
                    )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator