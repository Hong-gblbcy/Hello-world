#!/usr/bin/env python3
"""
安防平台API功能测试脚本
用于测试安防平台相关的API端点
"""

import requests
import json
import sys
from typing import Dict, Any, List

# API基础URL
BASE_URL = "http://localhost:8000/api/v1"

def get_auth_headers(token: str) -> Dict[str, str]:
    """获取认证头"""
    return {"Authorization": f"Bearer {token}"}

def login_user(username: str, password: str) -> str:
    """用户登录并返回token"""
    url = f"{BASE_URL}/auth/token"
    data = {
        "username": username,
        "password": password
    }
    
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            token = response.json()["access_token"]
            print(f"✅ 用户 {username} 登录成功")
            return token
        else:
            print(f"❌ 用户 {username} 登录失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 登录请求异常: {e}")
        return None

def create_device(token: str, project_id: int, device_data: Dict[str, Any]) -> Dict[str, Any]:
    """创建设备"""
    url = f"{BASE_URL}/projects/{project_id}/devices"
    headers = get_auth_headers(token)
    
    try:
        response = requests.post(url, json=device_data, headers=headers)
        if response.status_code == 201:
            device = response.json()
            print(f"✅ 设备创建成功: {device['name']} (ID: {device['id']})")
            return device
        else:
            print(f"❌ 设备创建失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 创建设备请求异常: {e}")
        return None

def get_device_status(token: str, project_id: int, index_code: str) -> Dict[str, Any]:
    """获取设备状态"""
    url = f"{BASE_URL}/projects/{project_id}/security/devices/{index_code}/status"
    headers = get_auth_headers(token)
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            status_info = response.json()
            print(f"✅ 获取设备状态成功: {index_code}")
            print(f"   状态: {status_info['status']}, 在线状态: {status_info['online_status']}")
            return status_info
        else:
            print(f"❌ 获取设备状态失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 获取设备状态请求异常: {e}")
        return None

def get_device_stream_url(token: str, project_id: int, index_code: str, protocol: str = "rtsp") -> Dict[str, Any]:
    """获取设备流媒体URL"""
    url = f"{BASE_URL}/projects/{project_id}/security/devices/{index_code}/stream"
    headers = get_auth_headers(token)
    params = {"protocol": protocol}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            stream_info = response.json()
            print(f"✅ 获取流媒体URL成功: {index_code}")
            print(f"   协议: {stream_info['protocol']}, URL: {stream_info['stream_url']}")
            return stream_info
        else:
            print(f"❌ 获取流媒体URL失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 获取流媒体URL请求异常: {e}")
        return None

def control_device(token: str, project_id: int, index_code: str, action: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    """控制设备"""
    url = f"{BASE_URL}/projects/{project_id}/security/devices/{index_code}/control"
    headers = get_auth_headers(token)
    data = {
        "action": action,
        "params": params or {}
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            control_result = response.json()
            print(f"✅ 设备控制成功: {index_code} -> {action}")
            print(f"   结果: {control_result['success']}, 消息: {control_result['message']}")
            return control_result
        else:
            print(f"❌ 设备控制失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 设备控制请求异常: {e}")
        return None

def get_alarms(token: str, project_id: int, **kwargs) -> List[Dict[str, Any]]:
    """获取报警事件"""
    url = f"{BASE_URL}/projects/{project_id}/security/alarms"
    headers = get_auth_headers(token)
    params = {k: v for k, v in kwargs.items() if v is not None}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            alarms = response.json()
            print(f"✅ 获取报警事件成功: 共 {len(alarms)} 条报警")
            for alarm in alarms[:3]:  # 只显示前3条
                print(f"   {alarm['alarm_type']} - {alarm['alarm_description']}")
            return alarms
        else:
            print(f"❌ 获取报警事件失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"❌ 获取报警事件请求异常: {e}")
        return []

def sync_devices(token: str, project_id: int, force_sync: bool = False) -> Dict[str, Any]:
    """同步设备"""
    url = f"{BASE_URL}/projects/{project_id}/security/sync-devices"
    headers = get_auth_headers(token)
    data = {"force_sync": force_sync}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 200:
            sync_result = response.json()
            print(f"✅ 设备同步成功")
            print(f"   总设备: {sync_result['total_devices']}, 新设备: {sync_result['new_devices']}")
            print(f"   更新设备: {sync_result['updated_devices']}, 失败设备: {sync_result['failed_devices']}")
            return sync_result
        else:
            print(f"❌ 设备同步失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 设备同步请求异常: {e}")
        return None

def get_platform_status(token: str) -> Dict[str, Any]:
    """获取平台状态"""
    url = f"{BASE_URL}/security/platform/status"
    headers = get_auth_headers(token)
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            status_info = response.json()
            print(f"✅ 获取平台状态成功")
            print(f"   平台: {status_info['platform_name']}, 状态: {status_info['status']}")
            print(f"   版本: {status_info['version']}, 连接状态: {status_info['connected']}")
            return status_info
        else:
            print(f"❌ 获取平台状态失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 获取平台状态请求异常: {e}")
        return None

def get_all_devices_status(token: str, project_id: int) -> List[Dict[str, Any]]:
    """获取所有设备状态"""
    url = f"{BASE_URL}/projects/{project_id}/security/devices/status"
    headers = get_auth_headers(token)
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            status_list = response.json()
            print(f"✅ 获取所有设备状态成功: 共 {len(status_list)} 个设备")
            
            online_count = sum(1 for device in status_list if device.get('online_status') == 'online')
            offline_count = sum(1 for device in status_list if device.get('online_status') == 'offline')
            unknown_count = sum(1 for device in status_list if device.get('online_status') not in ['online', 'offline'])
            
            print(f"   在线: {online_count}, 离线: {offline_count}, 未知: {unknown_count}")
            return status_list
        else:
            print(f"❌ 获取所有设备状态失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"❌ 获取所有设备状态请求异常: {e}")
        return []

def main():
    """主测试函数"""
    print("=" * 60)
    print("安防平台API功能测试")
    print("=" * 60)
    
    # 测试数据
    test_project_id = 1  # 假设项目ID为1
    admin_username = "admin"
    admin_password = "admin123"
    
    # 管理员登录
    print("\n1. 管理员登录")
    admin_token = login_user(admin_username, admin_password)
    if not admin_token:
        print("❌ 管理员登录失败，测试终止")
        return
    
    # 创建设备用于测试
    print("\n2. 创建设备用于测试")
    device_data = {
        "index_code": "CAM001",
        "name": "测试摄像头1",
        "ip_address": "192.168.1.100",
        "stream_url": "rtsp://192.168.1.100:554/stream",
        "status": "online",
        "project_id": test_project_id
    }
    
    device = create_device(admin_token, test_project_id, device_data)
    if not device:
        print("❌ 创建设备失败，测试终止")
        return
    
    index_code = device["index_code"]
    
    # 获取平台状态
    print("\n3. 获取安防平台状态")
    platform_status = get_platform_status(admin_token)
    
    # 获取设备状态
    print("\n4. 获取设备状态")
    device_status = get_device_status(admin_token, test_project_id, index_code)
    
    # 获取流媒体URL
    print("\n5. 获取流媒体URL")
    stream_url = get_device_stream_url(admin_token, test_project_id, index_code, "rtsp")
    
    # 设备控制（模拟）
    print("\n6. 设备控制测试")
    control_result = control_device(admin_token, test_project_id, index_code, "start", {"param1": "value1"})
    
    # 获取报警事件
    print("\n7. 获取报警事件")
    alarms = get_alarms(admin_token, test_project_id, alarm_type="motion", alarm_level="high")
    
    # 同步设备
    print("\n8. 同步设备")
    sync_result = sync_devices(admin_token, test_project_id, force_sync=False)
    
    # 获取所有设备状态
    print("\n9. 获取所有设备状态")
    all_status = get_all_devices_status(admin_token, test_project_id)
    
    print("\n" + "=" * 60)
    print("安防平台API功能测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()