#!/usr/bin/env python3
"""
项目管理功能测试脚本
"""
import sys
import os
import requests
import json
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(__file__))

BASE_URL = "http://localhost:8000/api/v1"

def get_auth_headers(token):
    """获取认证头"""
    return {"Authorization": f"Bearer {token}"}

def test_project_management():
    """测试项目管理功能"""
    print("=" * 50)
    print("项目管理功能测试")
    print("=" * 50)
    
    # 1. 用户登录
    print("\n1. 用户登录")
    login_data = {
        "username": "admin",
        "password": "admin123"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/auth/token", json=login_data)
        if response.status_code != 200:
            print(f"登录失败: {response.status_code}")
            print(response.json())
            return False
        
        auth_data = response.json()
        token = auth_data["access_token"]
        user_id = auth_data["user_id"]
        print(f"登录成功，用户ID: {user_id}")
        
    except requests.exceptions.ConnectionError:
        print("无法连接到服务器，请确保服务器正在运行")
        return False
    
    headers = get_auth_headers(token)
    
    # 2. 创建项目
    print("\n2. 创建项目")
    project_data = {
        "name": f"测试项目_{datetime.now().strftime('%H%M%S')}",
        "description": "这是一个测试项目"
    }
    
    response = requests.post(f"{BASE_URL}/projects", json=project_data, headers=headers)
    if response.status_code != 201:
        print(f"创建项目失败: {response.status_code}")
        print(response.json())
        return False
    
    project = response.json()
    project_id = project["id"]
    print(f"项目创建成功，ID: {project_id}")
    
    # 3. 获取用户项目列表
    print("\n3. 获取用户项目列表")
    response = requests.get(f"{BASE_URL}/projects", headers=headers)
    if response.status_code != 200:
        print(f"获取项目列表失败: {response.status_code}")
        print(response.json())
        return False
    
    projects_data = response.json()
    print(f"用户共有 {projects_data['total']} 个项目")
    for p in projects_data["projects"]:
        print(f"  - {p['name']} (ID: {p['id']})")
    
    # 4. 获取项目详情
    print(f"\n4. 获取项目详情 (ID: {project_id})")
    response = requests.get(f"{BASE_URL}/projects/{project_id}", headers=headers)
    if response.status_code != 200:
        print(f"获取项目详情失败: {response.status_code}")
        print(response.json())
        return False
    
    project_detail = response.json()
    print(f"项目名称: {project_detail['name']}")
    print(f"项目描述: {project_detail['description']}")
    print(f"创建者: {project_detail['created_by']}")
    
    # 5. 获取项目成员
    print(f"\n5. 获取项目成员 (ID: {project_id})")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/members", headers=headers)
    if response.status_code != 200:
        print(f"获取项目成员失败: {response.status_code}")
        print(response.json())
        return False
    
    members = response.json()
    print(f"项目共有 {len(members)} 个成员")
    for member in members:
        print(f"  - 用户ID: {member['user_id']}, 角色: {member['role']}")
    
    # 6. 获取当前用户在项目中的角色
    print(f"\n6. 获取当前用户在项目中的角色 (ID: {project_id})")
    response = requests.get(f"{BASE_URL}/projects/{project_id}/my-role", headers=headers)
    if response.status_code != 200:
        print(f"获取用户角色失败: {response.status_code}")
        print(response.json())
        return False
    
    role_data = response.json()
    print(f"当前用户在项目中的角色: {role_data['role']}")
    
    # 7. 更新项目信息
    print(f"\n7. 更新项目信息 (ID: {project_id})")
    update_data = {
        "name": f"更新后的测试项目_{datetime.now().strftime('%H%M%S')}",
        "description": "这是更新后的项目描述"
    }
    
    response = requests.put(f"{BASE_URL}/projects/{project_id}", json=update_data, headers=headers)
    if response.status_code != 200:
        print(f"更新项目失败: {response.status_code}")
        print(response.json())
        return False
    
    updated_project = response.json()
    print(f"项目更新成功: {updated_project['name']}")
    
    # 8. 创建测试用户用于添加成员
    print(f"\n8. 创建测试用户")
    test_user_data = {
        "username": f"testuser_{datetime.now().strftime('%H%M%S')}",
        "password": "test123",
        "email": f"test{datetime.now().strftime('%H%M%S')}@example.com",
        "full_name": "测试用户"
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=test_user_data)
    if response.status_code != 201:
        print(f"创建测试用户失败: {response.status_code}")
        print(response.json())
        # 继续测试其他功能
        print("跳过添加成员测试")
    else:
        test_user = response.json()
        test_user_id = test_user["id"]
        print(f"测试用户创建成功，ID: {test_user_id}")
        
        # 9. 添加项目成员
        print(f"\n9. 添加项目成员 (项目ID: {project_id}, 用户ID: {test_user_id})")
        member_data = {
            "user_id": test_user_id,
            "role": "member"
        }
        
        response = requests.post(f"{BASE_URL}/projects/{project_id}/members", json=member_data, headers=headers)
        if response.status_code != 201:
            print(f"添加项目成员失败: {response.status_code}")
            print(response.json())
        else:
            member_info = response.json()
            print(f"成员添加成功，角色: {member_info['role']}")
    
    # 10. 删除项目
    print(f"\n10. 删除项目 (ID: {project_id})")
    response = requests.delete(f"{BASE_URL}/projects/{project_id}", headers=headers)
    if response.status_code != 204:
        print(f"删除项目失败: {response.status_code}")
        print(response.json())
        return False
    
    print("项目删除成功")
    
    print("\n" + "=" * 50)
    print("项目管理功能测试完成！")
    print("=" * 50)
    return True

if __name__ == "__main__":
    test_project_management()