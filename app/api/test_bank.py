"""独立试题库 API — 教师/管理员上传选择题/填空题/简答题（含答案），用于组卷。"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_teacher_or_admin
from app.models import TestQuestion

router = APIRouter(prefix="/api/test-bank", tags=["试题库"])

QUESTION_TYPES = ("choice", "fill", "short_answer")


# ── Pydantic ──

class TestQuestionCreate(BaseModel):
    subject_id: int | None = None
    chapter: str | None = None
    kp_id: str | None = None
    question_type: str = Field(..., description="choice | fill | short_answer")
    question_text: str = Field(..., min_length=1)
    options: list | None = None          # choice: [{"key":"A","text":"..."}]
    answer_text: str = Field(..., min_length=1)
    difficulty: int = Field(default=3, ge=1, le=5)
    images: list | None = None


class TestQuestionUpdate(BaseModel):
    subject_id: int | None = None
    chapter: str | None = None
    kp_id: str | None = None
    question_type: str | None = None
    question_text: str | None = None
    options: list | None = None
    answer_text: str | None = None
    difficulty: int | None = Field(default=None, ge=1, le=5)
    verified: bool | None = None


def _to_out(q: TestQuestion) -> dict:
    return {
        "id": q.id,
        "subject_id": q.subject_id,
        "chapter": q.chapter,
        "kp_id": q.kp_id,
        "question_type": q.question_type,
        "question_text": q.question_text,
        "options": q.options,
        "answer_text": q.answer_text,
        "difficulty": q.difficulty,
        "images": q.images,
        "created_by": q.created_by,
        "verified": q.verified,
        "created_at": q.created_at.isoformat() if q.created_at else None,
    }


def _validate(body: TestQuestionCreate | TestQuestionUpdate):
    qtype = body.question_type
    if qtype is not None and qtype not in QUESTION_TYPES:
        raise HTTPException(400, f"无效题型：{qtype}（可选 choice/fill/short_answer）")
    if qtype == "choice" and getattr(body, "options", None) is None and not isinstance(body, TestQuestionUpdate):
        raise HTTPException(400, "选择题必须提供 options 选项列表")


# ── Routes ──

@router.get("")
async def list_questions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    subject_id: int | None = None,
    chapter: str | None = None,
    question_type: str | None = None,
    difficulty: int | None = None,
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    conditions = []
    if subject_id:
        conditions.append(TestQuestion.subject_id == subject_id)
    if chapter:
        conditions.append(TestQuestion.chapter == chapter)
    if question_type:
        conditions.append(TestQuestion.question_type == question_type)
    if difficulty:
        conditions.append(TestQuestion.difficulty == difficulty)
    if search:
        conditions.append(TestQuestion.question_text.ilike(f"%{search}%"))

    base = select(TestQuestion)
    if conditions:
        base = base.where(and_(*conditions))

    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    result = await db.execute(
        base.order_by(TestQuestion.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "data": [_to_out(q) for q in result.scalars().all()],
        "total": total or 0,
        "page": page,
        "page_size": page_size,
    }


@router.post("")
async def create_question(
    body: TestQuestionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    _validate(body)
    q = TestQuestion(
        subject_id=body.subject_id,
        chapter=body.chapter,
        kp_id=body.kp_id,
        question_type=body.question_type,
        question_text=body.question_text,
        options=body.options,
        answer_text=body.answer_text,
        difficulty=body.difficulty,
        images=body.images,
        created_by=current_user["user_id"],
        verified=False,
    )
    db.add(q)
    await db.commit()
    await db.refresh(q)
    return _to_out(q)


@router.put("/{question_id}")
async def update_question(
    question_id: int,
    body: TestQuestionUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    q = (await db.execute(select(TestQuestion).where(TestQuestion.id == question_id))).scalar_one_or_none()
    if not q:
        raise HTTPException(404, "题目不存在")

    if body.question_type is not None and body.question_type not in QUESTION_TYPES:
        raise HTTPException(400, f"无效题型：{body.question_type}")

    data = body.model_dump(exclude_unset=True)
    for key, value in data.items():
        setattr(q, key, value)
    await db.commit()
    await db.refresh(q)
    return _to_out(q)


@router.delete("/{question_id}")
async def delete_question(
    question_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    q = (await db.execute(select(TestQuestion).where(TestQuestion.id == question_id))).scalar_one_or_none()
    if not q:
        raise HTTPException(404, "题目不存在")
    await db.delete(q)
    await db.commit()
    return {"message": "题目已删除"}
