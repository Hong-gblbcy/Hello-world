from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

# 创建数据库引擎
engine = create_engine(
    settings.DATABASE_URL, 
    connect_args={"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}
)

# 创建会话本地类
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 声明基类
Base = declarative_base()

def get_db():
    """
    获取数据库会话
    
    Returns:
        Session: SQLAlchemy数据库会话
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """
    初始化数据库，创建所有表
    """
    from app.models.user import User
    from app.models.project import Project
    from app.models.device import Device
    # ProjectUser 模型在 project.py 中定义
    
    Base.metadata.create_all(bind=engine)
    
    # 创建默认超级用户
    db = SessionLocal()
    try:
        from app.services.auth import get_password_hash
        
        # 检查是否已存在超级用户
        existing_admin = db.query(User).filter(User.username == "admin").first()
        if not existing_admin:
            admin_user = User(
                username="admin",
                password_hash=get_password_hash("admin123"),
                email="admin@example.com",
                role="superuser"
            )
            db.add(admin_user)
            db.commit()
            print("默认超级用户创建成功: admin/admin123")
    except Exception as e:
        print(f"创建默认用户时出错: {e}")
        db.rollback()
    finally:
        db.close()