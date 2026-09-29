"""Subject management API — CRUD for academic subjects."""
import asyncio
import json
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

class SubjectQuestionsUpdate(BaseModel):
    questions: list[str]


def _parse_questions(raw: str | None) -> list[str]:
    """把数据库里的 JSON 字符串解析成问题列表。"""
    if not raw:
        return []
    try:
        data = json.loads(raw)
        return [str(q).strip() for q in data if str(q).strip()]
    except Exception:
        return []


def _subject_out(s: Subject) -> dict:
    return {
        "id": s.id,
        "name": s.name,
        "description": s.description,
        "example_questions": _parse_questions(s.example_questions),
        "is_active": s.is_active,
        "created_at": s.created_at.isoformat() if s.created_at else None,
    }


async def _generate_questions(name: str) -> list[str]:
    """用 LLM 为学科生成欢迎页示例问题（创建学科时一次性生成）。"""
    from app.services.llm_service import LLMService

    prompt = (
        f"你是一名数学教材编辑。请为「{name}」这门课程设计 4 个典型问题，"
        "作为答疑助手欢迎页的示例问题。\n"
        "要求：\n"
        "1. 每个问题一句话，20 字以内；\n"
        "2. 覆盖该课程最核心、最常见的知识点；\n"
        "3. 严格只输出 JSON 数组，例如 [\"问题1\",\"问题2\",\"问题3\",\"问题4\"]，不要输出其它内容。"
    )
    try:
        llm = LLMService()
        text = await asyncio.to_thread(llm.get_sync_response, prompt, 300, 0.3)
        s = text.find("[")
        e = text.rfind("]")
        if s != -1 and e != -1 and e > s:
            arr = json.loads(text[s:e + 1])
            return [str(q).strip() for q in arr if str(q).strip()][:4]
    except Exception as ex:
        logger.warning("为学科「%s」生成示例问题失败: %s", name, ex)
    return []


@router.get("/")
async def list_subjects(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List all active subjects."""
    result = await db.execute(select(Subject).where(Subject.is_active == True).order_by(Subject.name))
    subjects = result.scalars().all()
    return [_subject_out(s) for s in subjects]


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
            return {**_subject_out(existing), "message": "学科已恢复"}
        raise HTTPException(409, "学科已存在")

    questions = await _generate_questions(data.name)
    subject = Subject(
        name=data.name,
        description=data.description,
        example_questions=json.dumps(questions, ensure_ascii=False) if questions else None,
    )
    db.add(subject)
    await db.commit()
    await db.refresh(subject)
    return _subject_out(subject)


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


@router.put("/{subject_id}/questions")
async def update_subject_questions(
    subject_id: int,
    data: SubjectQuestionsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """更新学科的欢迎页示例问题（管理员）。"""
    subject = (await db.execute(select(Subject).where(Subject.id == subject_id))).scalar_one_or_none()
    if not subject:
        raise HTTPException(404, "学科不存在")
    questions = [q.strip() for q in data.questions if q.strip()]
    subject.example_questions = json.dumps(questions, ensure_ascii=False) if questions else None
    await db.commit()
    return {"message": "示例问题已更新", "example_questions": questions}


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
