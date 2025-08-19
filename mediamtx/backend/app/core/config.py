from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    # 项目配置
    PROJECT_NAME: str = "MediaMTX"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # 数据库配置
    DATABASE_URL: str = Field(
        default="sqlite:///./app.db", 
        env="DATABASE_URL"
    )
    
    # JWT配置
    SECRET_KEY: str = Field(
        default="your-secret-key-change-in-production", 
        env="SECRET_KEY"
    )
    ALGORITHM: str = Field(default="HS256", env="ALGORITHM")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30, 
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    
    # 服务器配置
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=8000, env="PORT")
    
    # CORS配置
    CORS_ORIGINS: list = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"],
        env="CORS_ORIGINS"
    )
    
    # 文件上传配置
    MAX_FILE_SIZE: int = Field(default=10 * 1024 * 1024, env="MAX_FILE_SIZE")  # 10MB
    
    # 安防平台配置
    SECURITY_PLATFORM_BASE_URL: str = Field(
        default="http://localhost:8080",
        env="SECURITY_PLATFORM_BASE_URL"
    )
    SECURITY_PLATFORM_USERNAME: str = Field(
        default="admin",
        env="SECURITY_PLATFORM_USERNAME"
    )
    SECURITY_PLATFORM_PASSWORD: str = Field(
        default="password",
        env="SECURITY_PLATFORM_PASSWORD"
    )
    SECURITY_PLATFORM_TIMEOUT: int = Field(
        default=30,
        env="SECURITY_PLATFORM_TIMEOUT"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# 全局配置实例
settings = Settings()

def get_settings() -> Settings:
    return settings