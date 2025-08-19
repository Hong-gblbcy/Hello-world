import requests
import json

BASE_URL = "http://localhost:8000"
API_BASE_URL = "http://localhost:8000/api/v1"

def test_health_check():
    """测试健康检查端点"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"健康检查: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"健康检查失败: {e}")
        return False

def test_root():
    """测试根端点"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"根端点: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"根端点测试失败: {e}")
        return False

def test_register_user():
    """测试用户注册"""
    try:
        import time
        timestamp = int(time.time())
        user_data = {
            "username": f"testuser_{timestamp}",
            "email": f"test_{timestamp}@example.com",
            "password": "testpassword123",
            "role": "user"
        }
        response = requests.post(f"{API_BASE_URL}/auth/register", json=user_data)
        print(f"用户注册: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"用户注册测试失败: {e}")
        return False

def test_login():
    """测试用户登录"""
    try:
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        response = requests.post(f"{API_BASE_URL}/auth/login", data=login_data)
        print(f"用户登录: {response.status_code}")
        if response.status_code == 200:
            token_data = response.json()
            print(f"获取到Token: {token_data['access_token'][:20]}...")
            return token_data['access_token']
        return None
    except Exception as e:
        print(f"用户登录测试失败: {e}")
        return None

def test_admin_endpoints(token):
    """测试管理员端点（需要超级用户权限）"""
    if not token:
        print("没有token，跳过管理员端点测试")
        return False
    
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        # 测试获取用户列表
        response = requests.get(f"{API_BASE_URL}/admin/users", headers=headers)
        print(f"获取用户列表: {response.status_code}")
        if response.status_code == 200:
            users = response.json()
            print(f"用户数量: {len(users)}")
        return response.status_code in [200, 403]  # 403表示权限不足，也是正常情况
    except Exception as e:
        print(f"管理员端点测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("开始测试API端点...")
    print("=" * 50)
    
    tests = [
        ("健康检查", test_health_check),
        ("根端点", test_root),
        ("用户注册", test_register_user),
        ("用户登录", test_login),
    ]
    
    passed = 0
    total = len(tests)
    token = None
    
    for test_name, test_func in tests:
        print(f"\n测试: {test_name}")
        result = test_func()
        if result:
            passed += 1
            print(f"✓ {test_name} 通过")
        else:
            print(f"✗ {test_name} 失败")
        
        # 如果是登录测试，保存token
        if test_name == "用户登录" and result:
            token = result
    
    # 如果有token，测试管理员端点
    if token:
        print(f"\n测试: 管理员端点")
        result = test_admin_endpoints(token)
        if result:
            passed += 1
            print("✓ 管理员端点测试通过")
        else:
            print("✗ 管理员端点测试失败")
        total += 1
    
    print("=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！API服务运行正常。")
        print(f"\nAPI文档地址: http://localhost:8000/docs")
    else:
        print("⚠️  部分测试失败，请检查服务状态。")

if __name__ == "__main__":
    main()