# 部署配置文档

## 开发环境部署

### 1. 环境要求
- Python 3.8+
- Node.js 14+
- SQLite3

### 2. 后端部署步骤

#### 创建虚拟环境
```bash
# 创建项目目录
mkdir mediamtx
cd mediamtx

# 创建Python虚拟环境
python -m venv venv

# 激活虚拟环境
# Linux/Mac
source venv/bin/activate
# Windows
venv\Scripts\activate
```

#### 安装依赖
```bash
# 安装后端依赖
pip install fastapi uvicorn sqlalchemy python-jose[cryptography] passlib python-multipart
```

#### 启动后端服务
```bash
# 开发模式启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或者使用生产模式
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. 前端部署步骤

#### 创建React项目
```bash
# 在项目根目录创建前端目录
mkdir frontend
cd frontend

# 创建React项目
npx create-react-app .
```

#### 安装必要依赖
```bash
# 安装UI组件库和路由
npm install antd @ant-design/icons axios react-router-dom
```

#### 启动前端服务
```bash
# 开发模式启动
npm start

# 构建生产版本
npm run build
```

## 生产环境部署

### 1. 使用Docker部署（推荐）

#### Dockerfile (后端)
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Dockerfile (前端)
```dockerfile
FROM node:16-alpine as build

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
```

#### docker-compose.yml
```yaml
version: '3.8'

services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=sqlite:///./app.db
    volumes:
      - ./data:/app/data

  frontend:
    build: ./frontend
    ports:
      - "3000:80"
    depends_on:
      - backend

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    depends_on:
      - backend
      - frontend
```

### 2. 传统部署方式

#### 后端部署
```bash
# 安装系统依赖
sudo apt update
sudo apt install python3-pip python3-venv nginx

# 配置系统服务
sudo nano /etc/systemd/system/mediamtx.service
```

#### 系统服务配置
```ini
[Unit]
Description=MediaMTX Backend Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/mediamtx/backend
Environment=PYTHONPATH=/opt/mediamtx/backend
ExecStart=/opt/mediamtx/backend/venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

#### Nginx配置
```nginx
server {
    listen 80;
    server_name your-domain.com;

    # 前端静态文件
    location / {
        root /opt/mediamtx/frontend/build;
        index index.html;
        try_files $uri $uri/ /index.html;
    }

    # API代理
    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # 静态文件缓存
    location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## 环境变量配置

### 后端环境变量
```bash
# 数据库配置
DATABASE_URL=sqlite:///./app.db
# 或者 MySQL
DATABASE_URL=mysql://user:password@localhost/mediamtx

# JWT配置
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# 服务器配置
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

### 前端环境变量
```javascript
// .env
REACT_APP_API_BASE_URL=http://localhost:8000/api
REACT_APP_APP_NAME=MediaMTX
```

## 监控与日志

### 日志配置
```python
# logging.conf
[loggers]
keys=root,uvicorn

[handlers]
keys=console,file

[formatters]
keys=default

[logger_root]
level=INFO
handlers=console,file

[handler_console]
class=StreamHandler
level=INFO
formatter=default
args=(sys.stdout,)

[handler_file]
class=FileHandler
level=INFO
formatter=default
args=('app.log', 'a')

[formatter_default]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

### 健康检查端点
```bash
# 健康检查
curl http://localhost:8000/health

# 性能监控
curl http://localhost:8000/metrics
```

## 备份与恢复

### 数据库备份
```bash
# SQLite备份
sqlite3 app.db .dump > backup_$(date +%Y%m%d).sql

# 自动备份脚本
0 2 * * * /usr/bin/sqlite3 /app/data/app.db .dump > /backup/mediamtx_$(date +\%Y\%m\%d).sql
```

### 恢复数据库
```bash
sqlite3 app.db < backup_20231201.sql
```

## 安全建议
1. 使用HTTPS加密传输
2. 定期更新依赖包
3. 配置防火墙规则
4. 使用强密码策略
5. 定期备份数据
6. 监控系统日志