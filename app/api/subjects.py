"""Subject management API — CRUD for academic subjects."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_admin_user, get_current_user, get_teacher_or_admin
from app.models import Subject

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/subjects", tags=["学科管理"])


class SubjectCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str = ""

class SubjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("/")
async def list_subjects(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all active subjects."""
    result = await db.execute(select(Subject).where(Subject.is_active == True).order_by(Subject.name))
    subjects = result.scalars().all()
    return [{"id": s.id, "name": s.name, "description": s.description, "is_active": s.is_active,
             "created_at": s.created_at.isoformat() if s.created_at else None} for s in subjects]


@router.post("/")
async def create_subject(
    data: SubjectCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Create a new subject (teacher/admin only). Accepts JSON body: {"name":"...", "description":"..."}"""
    existing = (await db.execute(select(Subject).where(Subject.name == data.name))).scalar_one_or_none()
    if existing:
        if not existing.is_active:
            existing.is_active = True
            await db.commit()
            return {"id": existing.id, "name": existing.name, "message": "学科已恢复"}
        raise HTTPException(409, "学科已存在")

    subject = Subject(name=data.name, description=data.description)
    db.add(subject)
    await db.commit()
    await db.refresh(subject)
    return {"id": subject.id, "name": subject.name, "description": subject.description}


@router.put("/{subject_id}")
async def update_subject(
    subject_id: int,
    data: SubjectUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Update subject name/description (teacher/admin)."""
    subject = (await db.execute(select(Subject).where(Subject.id == subject_id))).scalar_one_or_none()
    if not subject:
        raise HTTPException(404, "学科不存在")
    if data.name is not None:
        subject.name = data.name
    if data.description is not None:
        subject.description = data.description
    await db.commit()
    return {"message": "学科已更新", "id": subject_id}


@router.delete("/{subject_id}")
async def delete_subject(
    subject_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Soft-delete a subject (admin only)."""
    subject = (await db.execute(select(Subject).where(Subject.id == subject_id))).scalar_one_or_none()
    if not subject:
        raise HTTPException(404, "学科不存在")
    subject.is_active = False
    await db.commit()
    return {"message": "学科已删除"}
