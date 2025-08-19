from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class DeviceStatusResponse(BaseModel):
    """设备状态响应"""
    index_code: str
    name: str
    status: str
    online_status: str
    last_online_time: Optional[datetime] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

class StreamURLResponse(BaseModel):
    """流媒体URL响应"""
    index_code: str
    name: str
    stream_url: str
    protocol: str
    expires_at: Optional[datetime] = None

class DeviceControlRequest(BaseModel):
    """设备控制请求"""
    action: str  # start, stop, reboot, etc.
    params: Optional[Dict[str, Any]] = None

class DeviceControlResponse(BaseModel):
    """设备控制响应"""
    index_code: str
    action: str
    success: bool
    message: str
    timestamp: datetime

class AlarmEvent(BaseModel):
    """报警事件"""
    alarm_id: str
    index_code: str
    alarm_type: str
    alarm_level: str
    alarm_time: datetime
    alarm_description: str
    alarm_status: str  # active, confirmed, cleared
    confirm_time: Optional[datetime] = None
    clear_time: Optional[datetime] = None

class AlarmQueryParams(BaseModel):
    """报警查询参数"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    alarm_type: Optional[str] = None
    alarm_level: Optional[str] = None
    alarm_status: Optional[str] = None
    index_code: Optional[str] = None

class SyncDevicesRequest(BaseModel):
    """同步设备请求"""
    force_sync: bool = False

class SyncDevicesResponse(BaseModel):
    """同步设备响应"""
    total_devices: int
    new_devices: int
    updated_devices: int
    failed_devices: int
    sync_time: datetime
    details: Optional[List[Dict[str, Any]]] = None

class PlatformStatusResponse(BaseModel):
    """平台状态响应"""
    platform_name: str
    status: str
    version: str
    connected: bool
    last_connection_time: Optional[datetime] = None
    error_message: Optional[str] = None