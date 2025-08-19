#!/usr/bin/env python3
"""
调试脚本：测试权限装饰器的问题
"""
import requests
import json

BASE_URL = "http://localhost:8000/api/v1"

def test_login():
    """测试登录获取token"""
    url = f"{BASE_URL}/auth/token"
    data = {
        "username": "admin",
        "password": "admin123"
    }
    response = requests.post(url, json=data)
    if response.status_code == 200:
        token_data = response.json()
        print(f"登录成功，Token: {token_data['access_token']}")
        return token_data['access_token']
    else:
        print(f"登录失败: {response.status_code}")
        print(response.text)
        return None

def test_get_project_members(token, project_id):
    """测试获取项目成员"""
    url = f"{BASE_URL}/projects/{project_id}/members"
    headers = {"Authorization": f"Bearer {token}"}
    
    print(f"测试URL: {url}")
    print(f"Headers: {headers}")
    
    response = requests.get(url, headers=headers)
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    return response

def main():
    # 获取token
    token = test_login()
    if not token:
        return
    
    # 测试获取项目成员（使用一个存在的项目ID）
    print("\n=== 测试获取项目成员 ===")
    response = test_get_project_members(token, 1)
    
    # 如果有422错误，检查错误详情
    if response.status_code == 422:
        try:
            error_data = response.json()
            print("\n=== 422错误详情 ===")
            print(json.dumps(error_data, indent=2, ensure_ascii=False))
        except:
            print("无法解析422错误响应")

if __name__ == "__main__":
    main()