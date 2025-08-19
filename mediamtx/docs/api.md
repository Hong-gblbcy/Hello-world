# API接口文档

## 基础信息
- **Base URL**: `http://localhost:8000/api/v1`
- **认证方式**: Bearer Token (JWT)
- **响应格式**: JSON

## 认证接口

### 用户登录
- **URL**: `/auth/login`
- **Method**: POST
- **认证**: 不需要

**请求体**:
```json
{
  "username": "string",
  "password": "string"
}
```

**响应**:
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "user": {
    "id": 1,
    "username": "admin",
    "email": "admin@example.com",
    "role": "superuser"
  }
}
```

### 用户注册
- **URL**: `/auth/register`
- **Method**: POST
- **认证**: 不需要

**请求体**:
```json
{
  "username": "string",
  "password": "string",
  "email": "string",
  "role": "user"
}
```

**响应**: 同登录响应

## 用户管理接口

### 获取当前用户信息
- **URL**: `/users/me`
- **Method**: GET
- **认证**: 需要

**响应**:
```json
{
  "id": 1,
  "username": "admin",
  "email": "admin@example.com",
  "role": "superuser",
  "created_at": "2024-01-01T00:00:00"
}
```

### 获取用户列表
- **URL**: `/users`
- **Method**: GET
- **认证**: 需要 (仅超级用户)

**响应**:
```json
{
  "users": [
    {
      "id": 1,
      "username": "admin",
      "email": "admin@example.com",
      "role": "superuser",
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 1
}
```

## 项目管理接口

### 创建项目
- **URL**: `/projects`
- **Method**: POST
- **认证**: 需要

**请求体**:
```json
{
  "name": "项目名称",
  "description": "项目描述"
}
```

**响应**:
```json
{
  "id": 1,
  "name": "项目名称",
  "description": "项目描述",
  "created_by": 1,
  "created_at": "2024-01-01T00:00:00"
}
```

### 获取项目列表
- **URL**: `/projects`
- **Method**: GET
- **认证**: 需要

**响应**:
```json
{
  "projects": [
    {
      "id": 1,
      "name": "项目名称",
      "description": "项目描述",
      "created_by": 1,
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 1
}
```

### 获取项目详情
- **URL**: `/projects/{project_id}`
- **Method**: GET
- **认证**: 需要

**响应**: 同创建项目响应

## 设备管理接口

### 创建设备
- **URL**: `/devices`
- **Method**: POST
- **认证**: 需要

**请求体**:
```json
{
  "index_code": "设备标识码",
  "name": "设备名称",
  "ip_address": "192.168.1.100",
  "project_id": 1
}
```

**响应**:
```json
{
  "id": 1,
  "index_code": "设备标识码",
  "name": "设备名称",
  "ip_address": "192.168.1.100",
  "status": "online",
  "project_id": 1,
  "created_at": "2024-01-01T00:00:00"
}
```

### 获取设备列表
- **URL**: `/devices`
- **Method**: GET
- **认证**: 需要

**查询参数**:
- `project_id`: 项目ID (可选)
- `page`: 页码 (默认1)
- `limit`: 每页数量 (默认10)

**响应**:
```json
{
  "devices": [
    {
      "id": 1,
      "index_code": "设备标识码",
      "name": "设备名称",
      "ip_address": "192.168.1.100",
      "status": "online",
      "project_id": 1,
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 1,
  "page": 1,
  "limit": 10
}
```

## 错误响应格式
```json
{
  "detail": "错误描述",
  "code": "错误代码"
}
```

## 常见错误代码
- `AUTH_INVALID_CREDENTIALS`: 认证失败
- `PERMISSION_DENIED`: 权限不足
- `RESOURCE_NOT_FOUND`: 资源不存在
- `VALIDATION_ERROR`: 参数验证失败