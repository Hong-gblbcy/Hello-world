from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from app.core.database import get_db
from app.services.auth import get_current_user
from app.services.security_platform import security_platform_service
from app.services.device import DeviceService
from app.schemas.security import (
    DeviceStatusResponse,
    StreamURLResponse,
    DeviceControlRequest,
    DeviceControlResponse,
    AlarmEvent,
    AlarmQueryParams,
    SyncDevicesRequest,
    SyncDevicesResponse,
    PlatformStatusResponse
)
from app.models.user import User
from app.decorators.permissions import require_project_member, require_project_admin

router = APIRouter()

@router.get("/projects/{project_id}/security/devices/{index_code}/status", response_model=DeviceStatusResponse)
@require_project_member("project_id")
async def get_device_status(
    project_id: int,
    index_code: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取设备状态（需要项目成员权限）"""
    try:
        # 首先验证设备存在且属于该项目
        device = DeviceService.get_device_by_index_code(db, project_id, index_code)
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="设备不存在"
            )
        
        # 获取设备状态
        status_info = await security_platform_service.get_device_status(index_code)
        return DeviceStatusResponse(
            index_code=index_code,
            name=device.name,
            **status_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取设备状态失败: {str(e)}"
        )

@router.get("/projects/{project_id}/security/devices/{index_code}/stream", response_model=StreamURLResponse)
@require_project_member("project_id")
async def get_device_stream_url(
    project_id: int,
    index_code: str,
    protocol: str = Query("rtsp", description="流媒体协议，支持 rtsp, rtmp, hls"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取设备流媒体URL（需要项目成员权限）"""
    try:
        # 首先验证设备存在且属于该项目
        device = DeviceService.get_device_by_index_code(db, project_id, index_code)
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="设备不存在"
            )
        
        # 获取流媒体URL
        stream_info = await security_platform_service.get_device_stream_url(index_code, protocol)
        return StreamURLResponse(
            index_code=index_code,
            name=device.name,
            **stream_info
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取流媒体URL失败: {str(e)}"
        )

@router.post("/projects/{project_id}/security/devices/{index_code}/control", response_model=DeviceControlResponse)
@require_project_admin("project_id")
async def control_device(
    project_id: int,
    index_code: str,
    control_request: DeviceControlRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """控制设备（需要项目管理员权限）"""
    try:
        # 首先验证设备存在且属于该项目
        device = DeviceService.get_device_by_index_code(db, project_id, index_code)
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="设备不存在"
            )
        
        # 执行设备控制
        result = await security_platform_service.control_device(
            index_code, 
            control_request.action, 
            control_request.params
        )
        return DeviceControlResponse(
            index_code=index_code,
            action=control_request.action,
            **result
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"设备控制失败: {str(e)}"
        )

@router.get("/projects/{project_id}/security/alarms", response_model=List[AlarmEvent])
@require_project_member("project_id")
async def get_alarms(
    project_id: int,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    alarm_type: Optional[str] = None,
    alarm_level: Optional[str] = None,
    alarm_status: Optional[str] = None,
    index_code: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取报警事件（需要项目成员权限）"""
    try:
        # 构建查询参数
        query_params = {
            "start_time": start_time,
            "end_time": end_time,
            "alarm_type": alarm_type,
            "alarm_level": alarm_level,
            "alarm_status": alarm_status,
            "index_code": index_code
        }
        
        # 如果指定了设备索引码，验证设备属于该项目
        if index_code:
            device = DeviceService.get_device_by_index_code(db, project_id, index_code)
            if not device:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="设备不存在"
                )
        
        # 获取报警事件
        alarms = await security_platform_service.get_alarms(query_params)
        return alarms
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取报警事件失败: {str(e)}"
        )

@router.post("/projects/{project_id}/security/sync-devices", response_model=SyncDevicesResponse)
@require_project_admin("project_id")
async def sync_devices(
    project_id: int,
    sync_request: SyncDevicesRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """同步设备（需要项目管理员权限）"""
    try:
        # 同步设备
        sync_result = await security_platform_service.sync_devices(project_id, sync_request.force_sync)
        return SyncDevicesResponse(**sync_result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"同步设备失败: {str(e)}"
        )

@router.get("/security/platform/status", response_model=PlatformStatusResponse)
async def get_platform_status(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取安防平台状态（需要登录）"""
    try:
        status_info = await security_platform_service.get_platform_status()
        return PlatformStatusResponse(**status_info)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取平台状态失败: {str(e)}"
        )

@router.get("/projects/{project_id}/security/devices/status", response_model=List[DeviceStatusResponse])
@require_project_member("project_id")
async def get_all_devices_status(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取项目所有设备状态（需要项目成员权限）"""
    try:
        # 获取项目所有设备
        devices = DeviceService.get_project_devices(db, project_id)
        
        # 批量获取设备状态
        status_list = []
        for device in devices:
            try:
                status_info = await security_platform_service.get_device_status(device.index_code)
                status_list.append(DeviceStatusResponse(
                    index_code=device.index_code,
                    name=device.name,
                    **status_info
                ))
            except Exception as e:
                # 单个设备状态获取失败，记录错误但继续处理其他设备
                status_list.append(DeviceStatusResponse(
                    index_code=device.index_code,
                    name=device.name,
                    status="unknown",
                    online_status="offline",
                    error_message=f"获取状态失败: {str(e)}"
                ))
        
        return status_list
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取设备状态列表失败: {str(e)}"
        )