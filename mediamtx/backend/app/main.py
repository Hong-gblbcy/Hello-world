from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.database import engine, Base
from app.api.v1 import auth, admin, projects, devices, security

# 创建数据库表
Base.metadata.create_all(bind=engine)

# 创建FastAPI应用
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="安防管理系统API",
    version=settings.PROJECT_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册API路由
app.include_router(auth.router, prefix="/api/v1")
app.include_router(admin.router, prefix="/api/v1")
app.include_router(projects.router, prefix="/api/v1")
app.include_router(devices.router, prefix="/api/v1")
app.include_router(security.router, prefix="/api/v1")

@app.get("/")
async def root():
    """根端点"""
    return {
        "message": "欢迎使用安防管理系统API",
        "version": settings.PROJECT_VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG
    )