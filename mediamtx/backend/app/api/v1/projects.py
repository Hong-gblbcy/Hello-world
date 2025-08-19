from fastapi import APIRouter, Depends, HTTPException, status, Response
from sqlalchemy.orm import Session
from typing import List

from app.core.database import get_db
from app.services.auth import get_current_user
from app.services.project import ProjectService
from app.schemas.project import (
    ProjectCreate, ProjectUpdate, ProjectResponse, 
    ProjectUserCreate, ProjectUserResponse, ProjectListResponse
)
from app.models.user import User
from app.decorators.permissions import require_project_admin, require_project_member

router = APIRouter()

@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建新项目"""
    try:
        project = ProjectService.create_project(db, project_data, current_user.id)
        return project
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"创建项目失败: {str(e)}"
        )

@router.get("/projects", response_model=ProjectListResponse)
async def get_user_projects(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户参与的所有项目"""
    try:
        projects = ProjectService.get_user_projects(db, current_user.id)
        return {
            "projects": projects,
            "total": len(projects)
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"获取项目列表失败: {str(e)}"
        )

@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """根据ID获取项目详情"""
    # 检查用户是否有权限访问该项目
    if not ProjectService.is_project_member(db, project_id, current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问此项目"
        )
    
    project = ProjectService.get_project(db, project_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="项目不存在"
        )
    
    return project

@router.put("/projects/{project_id}", response_model=ProjectResponse)
@require_project_admin("project_id")
async def update_project(
    project_id: int,
    project_data: ProjectUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新项目信息（需要项目管理员权限）"""
    try:
        project = ProjectService.update_project(db, project_id, project_data)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="项目不存在"
            )
        return project
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"更新项目失败: {str(e)}"
        )

@router.delete("/projects/{project_id}")
@require_project_admin("project_id")
async def delete_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Response:
    """删除项目（需要项目管理员权限）"""
    try:
        ProjectService.delete_project(db, project_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"删除项目失败: {str(e)}"
        )

@router.post("/projects/{project_id}/members", response_model=ProjectUserResponse, status_code=status.HTTP_201_CREATED)
@require_project_admin("project_id")
async def add_project_member(
    project_id: int,
    member_data: ProjectUserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """添加项目成员（需要项目管理员权限）"""
    try:
        # 检查项目是否存在
        project = ProjectService.get_project(db, project_id)
        if not project:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="项目不存在"
            )
        
        # 检查用户是否存在
        from app.services.user import UserService
        user = UserService.get_user_by_id(db, member_data.user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="用户不存在"
            )
        
        project_member = ProjectService.add_project_member(
            db, project_id, member_data.user_id, member_data.role
        )
        return project_member
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"添加项目成员失败: {str(e)}"
        )

@router.get("/projects/{project_id}/members", response_model=List[ProjectUserResponse])
@require_project_member("project_id")
async def get_project_members(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取项目所有成员（需要项目成员权限）"""
    try:
        members = ProjectService.get_project_members(db, project_id)
        return members
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"获取项目成员失败: {str(e)}"
        )

@router.delete("/projects/{project_id}/members/{user_id}")
@require_project_admin("project_id")
async def remove_project_member(
    project_id: int,
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
) -> Response:
    """移除项目成员（需要项目管理员权限）"""
    try:
        # 不能移除自己（项目创建者）
        if user_id == current_user.id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="不能移除自己"
            )
        
        ProjectService.remove_project_member(db, project_id, user_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"移除项目成员失败: {str(e)}"
        )

@router.get("/projects/{project_id}/my-role")
@require_project_member("project_id")
async def get_my_role_in_project(
    project_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取当前用户在项目中的角色"""
    role = ProjectService.get_user_role_in_project(db, project_id, current_user.id)
    return {"role": role}