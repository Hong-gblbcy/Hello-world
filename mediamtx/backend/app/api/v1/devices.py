from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.services.auth import get_current_user
from app.services.device import DeviceService
from app.schemas.device import (
    DeviceCreate, DeviceUpdate, DeviceResponse, DeviceListResponse
)
from app.models.user import User
from app.decorators.permissions import require_project_member, require_project_admin

router = APIRouter()

@router.post("/projects/{project_id}/devices", response_model=DeviceResponse, status_code=status.HTTP_201_CREATED)
@require_project_admin("project_id")
async def create_device(
    project_id: int,
    device_data: DeviceCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建设备（需要项目管理员权限）"""
    try:
        # 确保项目ID一致
        if device_data.project_id != project_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="项目ID不匹配"
            )
        
        device = DeviceService.create_device(db, device_data)
        return device
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建设备失败: {str(e)}"
        )

@router.get("/projects/{project_id}/devices", response_model=DeviceListResponse)
@require_project_member("project_id")
async def get_project_devices(
    project_id: int,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取项目的所有设备（需要项目成员权限）"""
    try:
        devices = DeviceService.get_project_devices(db, project_id, skip, limit)
        total = DeviceService.get_project_device_count(db, project_id)
        
        return {
            "devices": devices,
            "total": total,
            "page": skip // limit + 1 if limit > 0 else 1,
            "limit": limit
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"获取设备列表失败: {str(e)}"
        )

@router.get("/devices/{device_id}", response_model=DeviceResponse)
async def get_device(
    device_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """根据ID获取设备详情"""
    device = DeviceService.get_device(db, device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="设备不存在"
        )
    
    # 检查用户是否有权限访问该设备所属的项目
    from app.services.project import ProjectService
    if not ProjectService.is_project_member(db, device.project_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问此设备"
        )
    
    return device

@router.put("/devices/{device_id}", response_model=DeviceResponse)
async def update_device(
    device_id: int,
    device_data: DeviceUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新设备信息（需要项目管理员权限）"""
    device = DeviceService.get_device(db, device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="设备不存在"
        )
    
    # 检查用户是否是项目管理员
    from app.services.project import ProjectService
    if not ProjectService.is_project_admin(db, device.project_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要项目管理员权限"
        )
    
    try:
        updated_device = DeviceService.update_device(db, device_id, device_data)
        if not updated_device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="设备不存在"
            )
        return updated_device
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新设备失败: {str(e)}"
        )

@router.delete("/devices/{device_id}")
async def delete_device(
    device_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除设备（需要项目管理员权限）"""
    device = DeviceService.get_device(db, device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="设备不存在"
        )
    
    # 检查用户是否是项目管理员
    from app.services.project import ProjectService
    if not ProjectService.is_project_admin(db, device.project_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要项目管理员权限"
        )
    
    try:
        success = DeviceService.delete_device(db, device_id)
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="设备不存在"
            )
        return {"message": "设备删除成功"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"删除设备失败: {str(e)}"
        )

@router.patch("/devices/{device_id}/status")
async def update_device_status(
    device_id: int,
    status: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新设备状态（需要项目管理员权限）"""
    device = DeviceService.get_device(db, device_id)
    if not device:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="设备不存在"
        )
    
    # 检查用户是否是项目管理员
    from app.services.project import ProjectService
    if not ProjectService.is_project_admin(db, device.project_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要项目管理员权限"
        )
    
    try:
        updated_device = DeviceService.update_device_status(db, device_id, status)
        if not updated_device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="设备不存在"
            )
        return updated_device
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新设备状态失败: {str(e)}"
        )