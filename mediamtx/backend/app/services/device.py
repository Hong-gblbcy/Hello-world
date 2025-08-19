from sqlalchemy.orm import Session
from sqlalchemy import and_
from typing import List, Optional

from app.models.device import Device
from app.schemas.device import DeviceCreate, DeviceUpdate

class DeviceService:
    """设备服务类"""
    
    @staticmethod
    def create_device(db: Session, device_data: DeviceCreate) -> Device:
        """
        创建设备
        
        Args:
            db: 数据库会话
            device_data: 设备创建数据
            
        Returns:
            Device: 创建的设备对象
        """
        # 检查设备标识码是否已存在
        existing_device = db.query(Device).filter(
            Device.index_code == device_data.index_code,
            Device.project_id == device_data.project_id
        ).first()
        
        if existing_device:
            raise ValueError("设备标识码已存在")
        
        device = Device(
            index_code=device_data.index_code,
            name=device_data.name,
            ip_address=device_data.ip_address,
            stream_url=device_data.stream_url,
            status=device_data.status,
            project_id=device_data.project_id
        )
        
        db.add(device)
        db.commit()
        db.refresh(device)
        return device
    
    @staticmethod
    def get_device(db: Session, device_id: int) -> Optional[Device]:
        """
        根据ID获取设备
        
        Args:
            db: 数据库会话
            device_id: 设备ID
            
        Returns:
            Optional[Device]: 设备对象或None
        """
        return db.query(Device).filter(Device.id == device_id).first()
    
    @staticmethod
    def get_project_devices(db: Session, project_id: int, skip: int = 0, limit: int = 100) -> List[Device]:
        """
        获取项目的所有设备
        
        Args:
            db: 数据库会话
            project_id: 项目ID
            skip: 跳过记录数
            limit: 限制返回记录数
            
        Returns:
            List[Device]: 设备列表
        """
        return db.query(Device).filter(Device.project_id == project_id).offset(skip).limit(limit).all()
    
    @staticmethod
    def get_project_device_count(db: Session, project_id: int) -> int:
        """
        获取项目的设备数量
        
        Args:
            db: 数据库会话
            project_id: 项目ID
            
        Returns:
            int: 设备数量
        """
        return db.query(Device).filter(Device.project_id == project_id).count()
    
    @staticmethod
    def update_device(db: Session, device_id: int, device_data: DeviceUpdate) -> Optional[Device]:
        """
        更新设备信息
        
        Args:
            db: 数据库会话
            device_id: 设备ID
            device_data: 设备更新数据
            
        Returns:
            Optional[Device]: 更新后的设备对象或None
        """
        device = db.query(Device).filter(Device.id == device_id).first()
        if not device:
            return None
        
        # 更新字段
        if device_data.name is not None:
            device.name = device_data.name
        if device_data.ip_address is not None:
            device.ip_address = device_data.ip_address
        if device_data.stream_url is not None:
            device.stream_url = device_data.stream_url
        if device_data.status is not None:
            device.status = device_data.status
        
        db.commit()
        db.refresh(device)
        return device
    
    @staticmethod
    def delete_device(db: Session, device_id: int) -> bool:
        """
        删除设备
        
        Args:
            db: 数据库会话
            device_id: 设备ID
            
        Returns:
            bool: 是否删除成功
        """
        device = db.query(Device).filter(Device.id == device_id).first()
        if not device:
            return False
        
        db.delete(device)
        db.commit()
        return True
    
    @staticmethod
    def get_device_by_index_code(db: Session, project_id: int, index_code: str) -> Optional[Device]:
        """
        根据设备标识码获取设备
        
        Args:
            db: 数据库会话
            project_id: 项目ID
            index_code: 设备标识码
            
        Returns:
            Optional[Device]: 设备对象或None
        """
        return db.query(Device).filter(
            and_(
                Device.project_id == project_id,
                Device.index_code == index_code
            )
        ).first()
    
    @staticmethod
    def update_device_status(db: Session, device_id: int, status: str) -> Optional[Device]:
        """
        更新设备状态
        
        Args:
            db: 数据库会话
            device_id: 设备ID
            status: 设备状态
            
        Returns:
            Optional[Device]: 更新后的设备对象或None
        """
        if status not in ['online', 'offline', 'maintenance']:
            raise ValueError("状态必须是 online、offline 或 maintenance")
        
        device = db.query(Device).filter(Device.id == device_id).first()
        if not device:
            return None
        
        device.status = status
        db.commit()
        db.refresh(device)
        return device