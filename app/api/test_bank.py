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
        "original_answer": q.original_answer,
        "llm_corrected": q.llm_corrected,
        "llm_verified": q.llm_verified,
        "difficulty": q.difficulty,
        "images": q.images,
        "created_by": q.created_by,
        "verified": q.verified,
        "source": q.source,
        "source_doc_id": q.source_doc_id,
        "page_number": q.page_number,
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
    kp_id: str | None = None,
    question_type: str | None = None,
    difficulty: int | None = None,
    source: str | None = None,
    search: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    conditions = []
    if subject_id:
        conditions.append(TestQuestion.subject_id == subject_id)
    if chapter:
        conditions.append(TestQuestion.chapter == chapter)
    if kp_id:
        conditions.append(TestQuestion.kp_id == kp_id)
    if question_type:
        conditions.append(TestQuestion.question_type == question_type)
    if difficulty:
        conditions.append(TestQuestion.difficulty == difficulty)
    if source:
        conditions.append(TestQuestion.source == source)
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


# ── 从解析文档抽取题目 ──

class ExtractRequest(BaseModel):
    doc_id: int = Field(..., description="已解析文档 ID")
    subject_id: int | None = Field(default=None, description="缺省时抽该文档所有 in_qb 学科")
    chapter: str | None = Field(default=None, description="可选：指定章名，整批题归入该章（无结构纯题目列表时用）")


@router.post("/extract")
async def extract_questions(
    body: ExtractRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """从解析文档抽取题目到试题库（题解分离 + 章/节对应），供组卷使用。"""
    from app.models import DocumentSubject
    from app.services.question_extractor import extract_questions_to_bank

    if body.subject_id:
        res = await extract_questions_to_bank(db, body.doc_id, body.subject_id, chapter_override=body.chapter)
        await db.commit()
        return res

    rows = (await db.execute(
        select(DocumentSubject).where(
            DocumentSubject.document_id == body.doc_id,
            DocumentSubject.in_qb.is_(True),
        )
    )).scalars().all()
    if not rows:
        raise HTTPException(404, "该文档未分配到任何学科的题库（in_qb）")

    results = []
    for r in rows:
        results.append(await extract_questions_to_bank(db, body.doc_id, r.subject_id, chapter_override=body.chapter))
    await db.commit()
    return {"doc_id": body.doc_id, "subjects": results}


@router.post("/rebuild")
async def rebuild_question_bank(
    subject_id: int = Query(..., description="学科 ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """重建某学科题库：对 in_qb 的文档增量抽题（含 LLM 校验 + 版本失效重抽）。"""
    from app.models import DocumentSubject
    from app.services.question_extractor import extract_questions_to_bank

    rows = (await db.execute(
        select(DocumentSubject).where(
            DocumentSubject.subject_id == subject_id,
            DocumentSubject.in_qb.is_(True),
        )
    )).scalars().all()
    if not rows:
        await db.commit()
        return {"subject_id": subject_id, "docs": [], "extracted": 0, "created": 0, "updated": 0, "kept": 0, "corrected": 0}

    doc_results = []
    created = updated = kept = corrected = 0
    for r in rows:
        res = await extract_questions_to_bank(db, r.document_id, r.subject_id, chapter_override=r.qb_chapter)
        created += res.get("created", 0)
        updated += res.get("updated", 0)
        kept += res.get("kept", 0)
        corrected += res.get("corrected", 0)
        doc_results.append({"doc_id": r.document_id, **res})
    await db.commit()
    return {
        "subject_id": subject_id,
        "docs": doc_results,
        "extracted": created + updated + kept,
        "created": created,
        "updated": updated,
        "kept": kept,
        "corrected": corrected,
    }
