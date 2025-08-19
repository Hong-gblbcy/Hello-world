from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class ProjectBase(BaseModel):
    """项目基础模型"""
    name: str = Field(..., min_length=2, max_length=100, description="项目名称")
    description: Optional[str] = Field(None, max_length=1000, description="项目描述")

class ProjectCreate(ProjectBase):
    """项目创建模型"""
    pass

class ProjectUpdate(BaseModel):
    """项目更新模型"""
    name: Optional[str] = Field(None, min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=1000)

class ProjectInDB(ProjectBase):
    """数据库中的项目模型"""
    id: int
    created_by: int
    created_at: datetime

    class Config:
        from_attributes = True

class ProjectResponse(ProjectInDB):
    """项目响应模型"""
    pass

class ProjectListResponse(BaseModel):
    """项目列表响应模型"""
    projects: list[ProjectResponse]
    total: int

class ProjectUserBase(BaseModel):
    """项目用户基础模型"""
    user_id: int = Field(..., description="用户ID")
    role: str = Field("member", description="用户在项目中的角色")

    @validator('role')
    def validate_role(cls, v):
        if v not in ['admin', 'member']:
            raise ValueError('角色必须是 admin 或 member')
        return v

class ProjectUserCreate(ProjectUserBase):
    """项目用户创建模型"""
    project_id: int = Field(..., description="项目ID")

class ProjectUserResponse(ProjectUserBase):
    """项目用户响应模型"""
    id: int
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True