from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """用户基础模型"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱地址")

class UserCreate(UserBase):
    """用户创建模型"""
    password: str = Field(..., min_length=6, max_length=100, description="密码")
    role: str = Field("user", description="用户角色")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['superuser', 'admin', 'user']:
            raise ValueError('角色必须是 superuser、admin 或 user')
        return v

class UserUpdate(BaseModel):
    """用户更新模型"""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    password: Optional[str] = Field(None, min_length=6, max_length=100)
    role: Optional[str] = Field(None, description="用户角色")

    @validator('role')
    def validate_role(cls, v):
        if v is not None and v not in ['superuser', 'admin', 'user']:
            raise ValueError('角色必须是 superuser、admin 或 user')
        return v

class UserInDB(UserBase):
    """数据库中的用户模型"""
    id: int
    role: str
    created_at: datetime

    class Config:
        from_attributes = True

class UserResponse(UserInDB):
    """用户响应模型"""
    pass

class Token(BaseModel):
    """Token响应模型"""
    access_token: str
    token_type: str = "bearer"
    user_id: Optional[int] = None

class TokenData(BaseModel):
    """Token数据模型"""
    username: Optional[str] = None

class LoginRequest(BaseModel):
    """登录请求模型"""
    username: str
    password: str