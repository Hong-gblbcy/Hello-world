#!/usr/bin/env python3
"""
设备管理功能测试脚本
用于测试设备相关的API端点
"""

import requests
import json
import sys
from typing import Dict, Any

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

def get_project_devices(token: str, project_id: int) -> list:
    """获取项目设备列表"""
    url = f"{BASE_URL}/projects/{project_id}/devices"
    headers = get_auth_headers(token)
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            result = response.json()
            devices = result["devices"]
            print(f"✅ 获取项目 {project_id} 的设备列表成功，共 {result['total']} 个设备")
            return devices
        else:
            print(f"❌ 获取设备列表失败: {response.status_code} - {response.text}")
            return []
    except Exception as e:
        print(f"❌ 获取设备列表请求异常: {e}")
        return []

def get_device(token: str, device_id: int) -> Dict[str, Any]:
    """获取设备详情"""
    url = f"{BASE_URL}/devices/{device_id}"
    headers = get_auth_headers(token)
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            device = response.json()
            print(f"✅ 获取设备详情成功: {device['name']}")
            return device
        else:
            print(f"❌ 获取设备详情失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 获取设备详情请求异常: {e}")
        return None

def update_device(token: str, device_id: int, update_data: Dict[str, Any]) -> Dict[str, Any]:
    """更新设备信息"""
    url = f"{BASE_URL}/devices/{device_id}"
    headers = get_auth_headers(token)
    
    try:
        response = requests.put(url, json=update_data, headers=headers)
        if response.status_code == 200:
            device = response.json()
            print(f"✅ 设备更新成功: {device['name']}")
            return device
        else:
            print(f"❌ 设备更新失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 更新设备请求异常: {e}")
        return None

def delete_device(token: str, device_id: int) -> bool:
    """删除设备"""
    url = f"{BASE_URL}/devices/{device_id}"
    headers = get_auth_headers(token)
    
    try:
        response = requests.delete(url, headers=headers)
        if response.status_code == 200:
            print(f"✅ 设备删除成功: {device_id}")
            return True
        else:
            print(f"❌ 设备删除失败: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ 删除设备请求异常: {e}")
        return False

def update_device_status(token: str, device_id: int, status: str) -> Dict[str, Any]:
    """更新设备状态"""
    url = f"{BASE_URL}/devices/{device_id}/status"
    headers = get_auth_headers(token)
    
    try:
        response = requests.patch(url, params={"status": status}, headers=headers)
        if response.status_code == 200:
            device = response.json()
            print(f"✅ 设备状态更新成功: {device['name']} -> {device['status']}")
            return device
        else:
            print(f"❌ 设备状态更新失败: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"❌ 更新设备状态请求异常: {e}")
        return None

def main():
    """主测试函数"""
    print("=" * 60)
    print("设备管理功能测试")
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
    
    # 创建设备
    print("\n2. 创建设备")
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
    
    device_id = device["id"]
    
    # 获取设备列表
    print("\n3. 获取项目设备列表")
    devices = get_project_devices(admin_token, test_project_id)
    print(f"设备列表: {[d['name'] for d in devices]}")
    
    # 获取设备详情
    print("\n4. 获取设备详情")
    device_detail = get_device(admin_token, device_id)
    
    # 更新设备信息
    print("\n5. 更新设备信息")
    update_data = {
        "name": "更新后的测试摄像头",
        "ip_address": "192.168.1.101",
        "stream_url": "rtsp://192.168.1.101:554/stream"
    }
    updated_device = update_device(admin_token, device_id, update_data)
    
    # 更新设备状态
    print("\n6. 更新设备状态")
    status_device = update_device_status(admin_token, device_id, "offline")
    
    # 再次获取设备详情确认更新
    print("\n7. 确认设备更新")
    final_device = get_device(admin_token, device_id)
    
    # 删除设备
    print("\n8. 删除设备")
    delete_success = delete_device(admin_token, device_id)
    
    # 最终确认设备已删除
    print("\n9. 确认设备已删除")
    deleted_device = get_device(admin_token, device_id)
    if deleted_device is None:
        print("✅ 设备删除确认成功")
    else:
        print("❌ 设备删除确认失败")
    
    print("\n" + "=" * 60)
    print("设备管理功能测试完成")
    print("=" * 60)

if __name__ == "__main__":
    main()