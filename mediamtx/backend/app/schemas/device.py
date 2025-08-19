from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class DeviceBase(BaseModel):
    """设备基础模型"""
    index_code: str = Field(..., min_length=1, max_length=100, description="设备标识码")
    name: str = Field(..., min_length=1, max_length=100, description="设备名称")
    ip_address: Optional[str] = Field(None, description="IP地址")
    stream_url: Optional[str] = Field(None, description="视频流地址")

class DeviceCreate(DeviceBase):
    """设备创建模型"""
    project_id: int = Field(..., description="项目ID")
    status: str = Field("online", description="设备状态")

    @validator('status')
    def validate_status(cls, v):
        if v not in ['online', 'offline', 'maintenance']:
            raise ValueError('状态必须是 online、offline 或 maintenance')
        return v

    @validator('ip_address')
    def validate_ip_address(cls, v):
        if v:
            # 简单的IP地址格式验证
            import re
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            if not re.match(ip_pattern, v):
                raise ValueError('IP地址格式不正确')
        return v

class DeviceUpdate(BaseModel):
    """设备更新模型"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    ip_address: Optional[str] = None
    stream_url: Optional[str] = None
    status: Optional[str] = None

    @validator('status')
    def validate_status(cls, v):
        if v and v not in ['online', 'offline', 'maintenance']:
            raise ValueError('状态必须是 online、offline 或 maintenance')
        return v

class DeviceInDB(DeviceBase):
    """数据库中的设备模型"""
    id: int
    status: str
    project_id: int
    created_at: datetime

    class Config:
        from_attributes = True

class DeviceResponse(DeviceInDB):
    """设备响应模型"""
    pass

class DeviceListResponse(BaseModel):
    """设备列表响应模型"""
    devices: list[DeviceResponse]
    total: int
    page: int
    limit: int