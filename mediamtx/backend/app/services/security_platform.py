"""
安防平台API集成服务
提供与外部安防平台（如海康威视、大华等）的API对接功能
"""

import requests
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)

class SecurityPlatformService:
    """安防平台服务类"""
    
    def __init__(self):
        self.base_url = settings.SECURITY_PLATFORM_BASE_URL
        self.username = settings.SECURITY_PLATFORM_USERNAME
        self.password = settings.SECURITY_PLATFORM_PASSWORD
        self.access_token = None
        self.token_expiry = None
    
    async def authenticate(self) -> bool:
        """认证到安防平台"""
        try:
            auth_url = f"{self.base_url}/api/v1/auth/login"
            auth_data = {
                "username": self.username,
                "password": self.password
            }
            
            response = requests.post(auth_url, json=auth_data, timeout=10)
            if response.status_code == 200:
                auth_result = response.json()
                self.access_token = auth_result.get("access_token")
                # 假设token有效期为1小时
                self.token_expiry = datetime.now().timestamp() + 3600
                logger.info("安防平台认证成功")
                return True
            else:
                logger.error(f"安防平台认证失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"安防平台认证异常: {e}")
            return False
    
    def _ensure_authenticated(self) -> bool:
        """确保已认证"""
        if not self.access_token or (self.token_expiry and datetime.now().timestamp() > self.token_expiry):
            return self.authenticate()
        return True
    
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Content-Type": "application/json"
        }
    
    async def get_device_status(self, device_index_code: str) -> Optional[Dict[str, Any]]:
        """获取设备状态"""
        if not self._ensure_authenticated():
            return None
            
        try:
            url = f"{self.base_url}/api/v1/devices/{device_index_code}/status"
            response = requests.get(url, headers=self._get_headers(), timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"获取设备状态失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"获取设备状态异常: {e}")
            return None
    
    async def get_device_stream_url(self, device_index_code: str, protocol: str = "rtsp") -> Optional[str]:
        """获取设备流地址"""
        if not self._ensure_authenticated():
            return None
            
        try:
            url = f"{self.base_url}/api/v1/devices/{device_index_code}/stream"
            params = {"protocol": protocol}
            response = requests.get(url, headers=self._get_headers(), params=params, timeout=10)
            
            if response.status_code == 200:
                stream_info = response.json()
                return stream_info.get("url")
            else:
                logger.warning(f"获取设备流地址失败: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"获取设备流地址异常: {e}")
            return None
    
    async def control_device(self, device_index_code: str, action: str, params: Dict[str, Any] = None) -> bool:
        """控制设备（如PTZ控制）"""
        if not self._ensure_authenticated():
            return False
            
        try:
            url = f"{self.base_url}/api/v1/devices/{device_index_code}/control"
            data = {
                "action": action,
                "params": params or {}
            }
            
            response = requests.post(url, json=data, headers=self._get_headers(), timeout=10)
            
            if response.status_code == 200:
                logger.info(f"设备控制成功: {device_index_code} - {action}")
                return True
            else:
                logger.warning(f"设备控制失败: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"设备控制异常: {e}")
            return False
    
    async def get_alarms(self, start_time: datetime, end_time: datetime, 
                        device_index_code: str = None, alarm_type: str = None) -> List[Dict[str, Any]]:
        """获取报警信息"""
        if not self._ensure_authenticated():
            return []
            
        try:
            url = f"{self.base_url}/api/v1/alarms"
            params = {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            if device_index_code:
                params["device_index_code"] = device_index_code
            if alarm_type:
                params["alarm_type"] = alarm_type
            
            response = requests.get(url, headers=self._get_headers(), params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json().get("alarms", [])
            else:
                logger.warning(f"获取报警信息失败: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"获取报警信息异常: {e}")
            return []
    
    async def sync_devices(self, project_id: int) -> Dict[str, Any]:
        """同步安防平台设备到本地数据库"""
        if not self._ensure_authenticated():
            return {"success": False, "message": "认证失败"}
            
        try:
            from app.services.device import DeviceService
            from app.core.database import SessionLocal
            from sqlalchemy.orm import Session
            
            db: Session = SessionLocal()
            
            # 获取安防平台设备列表
            url = f"{self.base_url}/api/v1/devices"
            response = requests.get(url, headers=self._get_headers(), timeout=30)
            
            if response.status_code != 200:
                return {"success": False, "message": f"获取设备列表失败: {response.status_code}"}
            
            platform_devices = response.json().get("devices", [])
            synced_count = 0
            updated_count = 0
            
            for device_data in platform_devices:
                # 检查设备是否已存在
                existing_device = DeviceService.get_device_by_index_code(db, device_data.get("index_code"))
                
                device_create_data = {
                    "index_code": device_data.get("index_code"),
                    "name": device_data.get("name"),
                    "ip_address": device_data.get("ip_address"),
                    "stream_url": device_data.get("stream_url"),
                    "status": device_data.get("status", "offline"),
                    "project_id": project_id
                }
                
                if existing_device:
                    # 更新现有设备
                    updated_device = DeviceService.update_device(
                        db, existing_device.id, device_create_data
                    )
                    if updated_device:
                        updated_count += 1
                else:
                    # 创建新设备
                    new_device = DeviceService.create_device(db, device_create_data)
                    if new_device:
                        synced_count += 1
            
            db.close()
            
            return {
                "success": True,
                "message": f"同步完成: 新增 {synced_count} 个设备, 更新 {updated_count} 个设备",
                "synced_count": synced_count,
                "updated_count": updated_count
            }
            
        except Exception as e:
            logger.error(f"同步设备异常: {e}")
            return {"success": False, "message": f"同步异常: {str(e)}"}

# 创建全局服务实例
security_platform_service = SecurityPlatformService()