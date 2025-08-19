from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import timedelta

from app.core.config import settings
from app.core.database import get_db
from app.services.auth import (
    authenticate_user, 
    create_access_token, 
    get_password_hash
)
from app.schemas.user import (
    UserCreate, 
    UserResponse, 
    Token, 
    LoginRequest
)
from app.models.user import User

router = APIRouter(prefix="/auth", tags=["认证"])

@router.post("/login", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """
    用户登录获取访问令牌
    
    Args:
        form_data: OAuth2密码请求表单
        db: 数据库会话
    
    Returns:
        Token: 访问令牌
    """
    user = authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """
    用户注册
    
    Args:
        user_data: 用户注册数据
        db: 数据库会话
    
    Returns:
        UserResponse: 注册成功的用户信息
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

@router.post("/token", response_model=Token)
async def login_for_access_token_json(
    login_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """
    JSON格式的用户登录（替代表单登录）
    
    Args:
        login_data: 登录请求数据
        db: 数据库会话
    
    Returns:
        Token: 访问令牌和用户信息
    """
    user = authenticate_user(db, login_data.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.id
    }