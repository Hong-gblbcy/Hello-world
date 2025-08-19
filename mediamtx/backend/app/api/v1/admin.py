from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional

from app.core.database import get_db
from app.services.auth import get_current_superuser
from app.models.user import User
from app.schemas.user import UserResponse, UserUpdate, UserCreate
from app.services.auth import get_password_hash

router = APIRouter(prefix="/admin", tags=["管理员"])

@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    获取用户列表（仅超级用户）
    
    Args:
        skip: 跳过记录数
        limit: 限制返回记录数
        current_user: 当前用户（依赖注入）
        db: 数据库会话
    
    Returns:
        List[UserResponse]: 用户列表
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return users

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    获取用户详情（仅超级用户）
    
    Args:
        user_id: 用户ID
        current_user: 当前用户（依赖注入）
        db: 数据库会话
    
    Returns:
        UserResponse: 用户详情
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    return user

@router.post("/users", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    创建用户（仅超级用户）
    
    Args:
        user_data: 用户创建数据
        current_user: 当前用户（依赖注入）
        db: 数据库会话
    
    Returns:
        UserResponse: 创建的用户信息
    """
    # 检查用户名是否已存在
    existing_user = db.query(User).filter(User.username == user_data.username).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 检查邮箱是否已存在
    existing_email = db.query(User).filter(User.email == user_data.email).first()
    if existing_email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已存在"
        )
    
    # 创建新用户
    hashed_password = get_password_hash(user_data.password)
    user = User(
        username=user_data.username,
        password_hash=hashed_password,
        email=user_data.email,
        role=user_data.role
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user

@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    更新用户信息（仅超级用户）
    
    Args:
        user_id: 用户ID
        user_data: 用户更新数据
        current_user: 当前用户（依赖注入）
        db: 数据库会话
    
    Returns:
        UserResponse: 更新后的用户信息
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 检查用户名是否冲突（排除当前用户）
    if user_data.username and user_data.username != user.username:
        existing_user = db.query(User).filter(
            User.username == user_data.username,
            User.id != user_id
        ).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="用户名已存在"
            )
    
    # 检查邮箱是否冲突（排除当前用户）
    if user_data.email and user_data.email != user.email:
        existing_email = db.query(User).filter(
            User.email == user_data.email,
            User.id != user_id
        ).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="邮箱已存在"
            )
    
    # 更新用户信息
    if user_data.username:
        user.username = user_data.username
    if user_data.email:
        user.email = user_data.email
    if user_data.role:
        user.role = user_data.role
    if user_data.password:
        user.password_hash = get_password_hash(user_data.password)
    
    db.commit()
    db.refresh(user)
    
    return user

@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(get_current_superuser),
    db: Session = Depends(get_db)
):
    """
    删除用户（仅超级用户）
    
    Args:
        user_id: 用户ID
        current_user: 当前用户（依赖注入）
        db: 数据库会话
    
    Returns:
        dict: 删除成功消息
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    # 不能删除自己
    if user.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="不能删除自己的账户"
        )
    
    db.delete(user)
    db.commit()
    
    return {"message": "用户删除成功"}