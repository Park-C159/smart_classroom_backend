"""组卷 API — 作业/测试/考试/练习：生成、发布、作答、判分、批改、掌握度联动。"""
import random
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user, get_teacher_or_admin
from app.models import (
    User, Paper, PaperQuestion, PaperSubmission, PaperAnswer, TestQuestion,
)
from app.services.llm_service import LLMService
from app.services.mastery_service import update_kp_mastery

router = APIRouter(prefix="/api/papers", tags=["组卷"])

VALID_MODES = ("practice", "homework", "test", "exam")
TYPE_ORDER = ("choice", "fill", "short_answer")
OBJECTIVE_TYPES = ("choice", "fill")


# ── Pydantic ──

class Counts(BaseModel):
    choice: int = 0
    fill: int = 0
    short_answer: int = 0


class GenerateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    mode: str = "practice"
    subject_id: int | None = None
    chapter: str | None = None
    difficulty_min: int = Field(default=1, ge=1, le=5)
    difficulty_max: int = Field(default=5, ge=1, le=5)
    counts: Counts = Field(default_factory=Counts)
    target_class: str | None = None
    due_at: datetime | None = None


class AnswerInput(BaseModel):
    question_id: int
    user_answer: str = ""


class SaveRequest(BaseModel):
    answers: list[AnswerInput] = []


class RegradeInput(BaseModel):
    answer_id: int
    score: float = Field(..., ge=0, le=1)
    feedback: str = ""


class RegradeRequest(BaseModel):
    answers: list[RegradeInput]


# ── Helpers ──

def _paper_to_out(p: Paper) -> dict:
    return {
        "id": p.id,
        "title": p.title,
        "subject_id": p.subject_id,
        "mode": p.mode,
        "chapter": p.chapter,
        "difficulty_min": p.difficulty_min,
        "difficulty_max": p.difficulty_max,
        "created_by": p.created_by,
        "target_class": p.target_class,
        "published": p.published,
        "due_at": p.due_at.isoformat() if p.due_at else None,
        "created_at": p.created_at.isoformat() if p.created_at else None,
    }


def _q_to_out(q: PaperQuestion, show_answer: bool = False) -> dict:
    out = {
        "id": q.id,
        "question_text": q.question_text,
        "question_type": q.question_type,
        "options": q.options,
        "difficulty": q.difficulty,
        "sort_order": q.sort_order,
    }
    if show_answer:
        out["answer_text"] = q.answer_text
    return out


async def _get_user_class(db, user_id: int) -> str | None:
    u = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    return u.class_name if u else None


async def _paper_visible_to(db, paper: Paper, user_id: int, role: str) -> bool:
    if role in ("teacher", "admin"):
        return paper.created_by == user_id or role == "admin"
    if not paper.published:
        return False
    if paper.mode == "practice":
        return paper.created_by == user_id
    cls = await _get_user_class(db, user_id)
    return bool(paper.target_class) and paper.target_class == cls


# ── Generate ──

@router.post("/generate")
async def generate_paper(
    body: GenerateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    mode = body.mode
    role = current_user["role"]
    if mode not in VALID_MODES:
        raise HTTPException(400, f"无效类型：{mode}")
    if mode in ("homework", "test", "exam"):
        if role not in ("teacher", "admin"):
            raise HTTPException(403, "只有教师或管理员可以创建作业/测试/考试")
        if not body.target_class:
            raise HTTPException(400, "作业/测试/考试需要指定班级")

    counts = body.counts
    selected: list[TestQuestion] = []
    for qtype in TYPE_ORDER:
        n = max(0, int(getattr(counts, qtype, 0) or 0))
        if n == 0:
            continue
        conds = [
            TestQuestion.question_type == qtype,
            TestQuestion.difficulty >= body.difficulty_min,
            TestQuestion.difficulty <= body.difficulty_max,
        ]
        if body.chapter:
            conds.append(TestQuestion.chapter == body.chapter)
        if body.subject_id:
            conds.append(TestQuestion.subject_id == body.subject_id)
        pool = (await db.execute(select(TestQuestion).where(and_(*conds)))).scalars().all()
        selected.extend(random.sample(list(pool), min(len(pool), n)) if pool else [])

    if not selected:
        raise HTTPException(400, "题库中没有符合条件的题目，请先上传题目或调整筛选条件")

    paper = Paper(
        title=body.title,
        subject_id=body.subject_id,
        mode=mode,
        chapter=body.chapter,
        difficulty_min=body.difficulty_min,
        difficulty_max=body.difficulty_max,
        created_by=current_user["user_id"],
        target_class=body.target_class if mode != "practice" else None,
        published=(mode == "practice"),
        due_at=body.due_at,
    )
    db.add(paper)
    await db.flush()

    for i, tq in enumerate(selected):
        db.add(PaperQuestion(
            paper_id=paper.id,
            test_question_id=tq.id,
            question_text=tq.question_text,
            question_type=tq.question_type,
            options=tq.options,
            answer_text=tq.answer_text,
            difficulty=tq.difficulty,
            kp_id=tq.kp_id,
            sort_order=i,
        ))
    await db.commit()

    result = await db.execute(
        select(Paper).where(Paper.id == paper.id).options(selectinload(Paper.questions))
    )
    paper = result.scalar_one()
    return {
        "paper": _paper_to_out(paper),
        "questions": [_q_to_out(q) for q in sorted(paper.questions, key=lambda x: x.sort_order)],
    }


# ── Publish ──

@router.post("/{paper_id}/publish")
async def publish_paper(
    paper_id: int,
    current_user: dict = Depends(get_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
):
    paper = (await db.execute(select(Paper).where(Paper.id == paper_id))).scalar_one_or_none()
    if not paper:
        raise HTTPException(404, "试卷不存在")
    if current_user["role"] != "admin" and paper.created_by != current_user["user_id"]:
        raise HTTPException(403, "无权操作该试卷")
    paper.published = True
    await db.commit()
    return _paper_to_out(paper)


# ── List ──

@router.get("")
async def list_papers(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    mode: str | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["user_id"]
    role = current_user["role"]
    conds = []
    if mode:
        conds.append(Paper.mode == mode)

    if role in ("teacher", "admin"):
        if role == "teacher":
            conds.append(Paper.created_by == user_id)
    else:
        # 学生：已发布 且（作业/测试/考试匹配班级 或 自测练习属于自己）
        cls = await _get_user_class(db, user_id)
        conds.append(Paper.published.is_(True))
        or_cond = []
        if cls:
            or_cond.append(Paper.target_class == cls)
        or_cond.append(and_(Paper.mode == "practice", Paper.created_by == user_id))
        from sqlalchemy import or_
        conds.append(or_(*or_cond))

    base = select(Paper)
    if conds:
        base = base.where(and_(*conds))
    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    result = await db.execute(
        base.order_by(Paper.created_at.desc()).offset((page - 1) * page_size).limit(page_size)
    )
    papers = result.scalars().all()

    items = []
    for p in papers:
        item = _paper_to_out(p)
        qcount = await db.scalar(
            select(func.count(PaperQuestion.id)).where(PaperQuestion.paper_id == p.id)
        )
        item["question_count"] = qcount or 0
        if role in ("teacher", "admin"):
            item["submission_count"] = await db.scalar(
                select(func.count(PaperSubmission.id)).where(PaperSubmission.paper_id == p.id)
            ) or 0
        else:
            sub = (await db.execute(
                select(PaperSubmission).where(PaperSubmission.paper_id == p.id, PaperSubmission.user_id == user_id)
            )).scalar_one_or_none()
            item["submission_id"] = sub.id if sub else None
            item["submission_status"] = sub.status if sub else None
            item["score"] = sub.score if sub else None
        items.append(item)

    return {"data": items, "total": total or 0, "page": page, "page_size": page_size}


# ── Get paper detail ──

@router.get("/{paper_id}")
async def get_paper(
    paper_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    paper = (await db.execute(
        select(Paper).where(Paper.id == paper_id).options(selectinload(Paper.questions))
    )).scalar_one_or_none()
    if not paper:
        raise HTTPException(404, "试卷不存在")

    role = current_user["role"]
    if not await _paper_visible_to(db, paper, current_user["user_id"], role):
        raise HTTPException(403, "无权查看该试卷")

    # 教师/管理员看答案；学生预览不看答案
    show_answer = role in ("teacher", "admin")
    return {
        "paper": _paper_to_out(paper),
        "questions": [_q_to_out(q, show_answer) for q in sorted(paper.questions, key=lambda x: x.sort_order)],
    }


# ── Start answering ──

@router.post("/{paper_id}/start")
async def start_paper(
    paper_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    user_id = current_user["user_id"]
    paper = (await db.execute(
        select(Paper).where(Paper.id == paper_id).options(selectinload(Paper.questions))
    )).scalar_one_or_none()
    if not paper:
        raise HTTPException(404, "试卷不存在")
    if not await _paper_visible_to(db, paper, user_id, current_user["role"]):
        raise HTTPException(403, "无权作答该试卷")

    sub = (await db.execute(
        select(PaperSubmission).where(PaperSubmission.paper_id == paper_id, PaperSubmission.user_id == user_id)
        .options(selectinload(PaperSubmission.answers))
    )).scalar_one_or_none()

    if sub:
        if sub.status == "submitted" and paper.mode in ("test", "exam"):
            raise HTTPException(403, "测试/考试提交后不可继续作答或更改")
        a_map = {a.paper_question_id: a.user_answer for a in sub.answers}
        questions_out = [
            {**_q_to_out(q), "user_answer": a_map.get(q.id, "") or ""}
            for q in sorted(paper.questions, key=lambda x: x.sort_order)
        ]
        return {
            "paper": _paper_to_out(paper),
            "submission": {"id": sub.id, "status": sub.status, "score": sub.score},
            "questions": questions_out,
        }

    sub = PaperSubmission(paper_id=paper_id, user_id=user_id, status="draft")
    db.add(sub)
    await db.flush()
    for q in sorted(paper.questions, key=lambda x: x.sort_order):
        db.add(PaperAnswer(submission_id=sub.id, paper_question_id=q.id))
    await db.commit()

    a_map = {a.paper_question_id: a.user_answer for a in sub.answers}
    questions_out = [
        {**_q_to_out(q), "user_answer": a_map.get(q.id, "") or ""}
        for q in sorted(paper.questions, key=lambda x: x.sort_order)
    ]
    return {
        "paper": _paper_to_out(paper),
        "submission": {"id": sub.id, "status": sub.status, "score": sub.score},
        "questions": questions_out,
    }


# ── Save draft ──

@router.post("/submissions/{sid}/save")
async def save_submission(
    sid: int,
    body: SaveRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sub = (await db.execute(
        select(PaperSubmission).where(PaperSubmission.id == sid).options(selectinload(PaperSubmission.answers))
    )).scalar_one_or_none()
    if not sub:
        raise HTTPException(404, "答卷不存在")
    if sub.user_id != current_user["user_id"]:
        raise HTTPException(403, "无权操作该答卷")

    paper = (await db.execute(select(Paper).where(Paper.id == sub.paper_id))).scalar_one()
    if sub.status == "submitted" and paper.mode in ("test", "exam"):
        raise HTTPException(403, "测试/考试提交后不可继续作答或更改")
    if sub.status == "submitted":
        sub.status = "draft"  # 作业/练习提交后仍可修改

    a_map = {a.paper_question_id: a for a in sub.answers}
    for ans in body.answers:
        a = a_map.get(ans.question_id)
        if a:
            a.user_answer = ans.user_answer
    await db.commit()
    return {"submission": {"id": sub.id, "status": sub.status}}


# ── Submit / grade ──

@router.post("/submissions/{sid}/submit")
async def submit_submission(
    sid: int,
    body: SaveRequest | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sub = (await db.execute(
        select(PaperSubmission).where(PaperSubmission.id == sid).options(selectinload(PaperSubmission.answers))
    )).scalar_one_or_none()
    if not sub:
        raise HTTPException(404, "答卷不存在")
    if sub.user_id != current_user["user_id"]:
        raise HTTPException(403, "无权操作该答卷")

    paper = (await db.execute(
        select(Paper).where(Paper.id == sub.paper_id).options(selectinload(Paper.questions))
    )).scalar_one()

    if sub.status == "submitted" and paper.mode in ("test", "exam"):
        raise HTTPException(403, "测试/考试提交后不可继续作答或更改")

    # 先保存本次提交携带的答案
    a_map = {a.paper_question_id: a for a in sub.answers}
    if body:
        for ans in body.answers:
            a = a_map.get(ans.question_id)
            if a:
                a.user_answer = ans.user_answer

    # 判分
    q_map = {q.id: q for q in paper.questions}
    llm = LLMService()
    total_score = 0.0
    correct = 0
    for a in sub.answers:
        q = q_map.get(a.paper_question_id)
        if not q:
            continue
        user_ans = (a.user_answer or "").strip()
        if q.question_type in OBJECTIVE_TYPES:
            ok = user_ans.lower() == (q.answer_text or "").strip().lower()
            a.is_correct = ok
            a.score = 1.0 if ok else 0.0
            a.graded_by = "auto"
            a.feedback = None
        else:  # short_answer
            grade = llm.grade_short_answer(q.question_text, q.answer_text or "", user_ans)
            if grade is not None:
                a.score = grade["score"]
                a.feedback = grade["feedback"]
                a.is_correct = a.score >= 0.5
                a.graded_by = "llm"
            else:
                ok = user_ans.lower() == (q.answer_text or "").strip().lower()
                a.is_correct = ok
                a.score = 1.0 if ok else 0.0
                a.graded_by = "auto"
        total_score += a.score or 0.0
        if a.score and a.score >= 0.5:
            correct += 1

    n = max(1, len(sub.answers))
    sub.score = round(total_score / n * 100, 1)
    sub.correct_count = correct
    sub.status = "submitted"
    sub.submitted_at = datetime.now(sub.created_at.tzinfo) if sub.created_at else datetime.now()

    # 逐题知识点掌握度
    kp_rates: dict[str, list[float]] = {}
    for a in sub.answers:
        q = q_map.get(a.paper_question_id)
        if q and q.kp_id:
            kp_rates.setdefault(q.kp_id, []).append(a.score if a.score is not None else 0.0)
    for kp_id, rates in kp_rates.items():
        rate = sum(rates) / len(rates) if rates else 0.0
        await update_kp_mastery(db, sub.user_id, kp_id, rate)

    await db.commit()

    return await _submission_out(sub, paper, viewer_role="student", db=db)


async def _submission_out(sub, paper, viewer_role: str, db) -> dict:
    q_map = {q.id: q for q in paper.questions}
    answers_out = []
    for a in sorted(sub.answers, key=lambda x: x.id):
        q = q_map.get(a.paper_question_id)
        item = {
            "answer_id": a.id,
            "question_id": a.paper_question_id,
            "question_text": q.question_text if q else "",
            "question_type": q.question_type if q else "",
            "options": q.options if q else None,
            "user_answer": a.user_answer,
            "is_correct": a.is_correct,
            "score": a.score,
            "feedback": a.feedback,
            "graded_by": a.graded_by,
        }
        if viewer_role in ("teacher", "admin"):
            item["answer_text"] = q.answer_text if q else None
        elif sub.status == "submitted":
            item["answer_text"] = q.answer_text if q else None
        answers_out.append(item)
    return {
        "paper": _paper_to_out(paper),
        "submission": {
            "id": sub.id,
            "status": sub.status,
            "score": sub.score,
            "correct_count": sub.correct_count,
            "submitted_at": sub.submitted_at.isoformat() if sub.submitted_at else None,
        },
        "answers": answers_out,
    }


# ── Get submission ──

@router.get("/submissions/{sid}")
async def get_submission(
    sid: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    sub = (await db.execute(
        select(PaperSubmission).where(PaperSubmission.id == sid).options(selectinload(PaperSubmission.answers))
    )).scalar_one_or_none()
    if not sub:
        raise HTTPException(404, "答卷不存在")

    role = current_user["role"]
    if role == "student" and sub.user_id != current_user["user_id"]:
        raise HTTPException(403, "无权查看该答卷")

    paper = (await db.execute(
        select(Paper).where(Paper.id == sub.paper_id).options(selectinload(Paper.questions))
    )).scalar_one()
    return await _submission_out(sub, paper, viewer_role=role, db=db)


# ── Teacher: list submissions of a paper ──

@router.get("/{paper_id}/submissions")
async def list_submissions(
    paper_id: int,
    current_user: dict = Depends(get_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
):
    paper = (await db.execute(select(Paper).where(Paper.id == paper_id))).scalar_one_or_none()
    if not paper:
        raise HTTPException(404, "试卷不存在")
    if current_user["role"] != "admin" and paper.created_by != current_user["user_id"]:
        raise HTTPException(403, "无权查看该试卷答卷")

    subs = (await db.execute(
        select(PaperSubmission).where(PaperSubmission.paper_id == paper_id)
        .order_by(PaperSubmission.created_at.desc())
    )).scalars().all()

    out = []
    for s in subs:
        u = (await db.execute(select(User).where(User.id == s.user_id))).scalar_one_or_none()
        out.append({
            "id": s.id,
            "user_id": s.user_id,
            "username": u.username if u else None,
            "real_name": u.real_name if u else None,
            "status": s.status,
            "score": s.score,
            "correct_count": s.correct_count,
            "submitted_at": s.submitted_at.isoformat() if s.submitted_at else None,
        })
    return {"paper": _paper_to_out(paper), "submissions": out}


# ── Teacher: regrade ──

@router.post("/submissions/{sid}/regrade")
async def regrade_submission(
    sid: int,
    body: RegradeRequest,
    current_user: dict = Depends(get_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
):
    sub = (await db.execute(
        select(PaperSubmission).where(PaperSubmission.id == sid).options(selectinload(PaperSubmission.answers))
    )).scalar_one_or_none()
    if not sub:
        raise HTTPException(404, "答卷不存在")
    paper = (await db.execute(select(Paper).where(Paper.id == sub.paper_id))).scalar_one()
    if current_user["role"] != "admin" and paper.created_by != current_user["user_id"]:
        raise HTTPException(403, "无权批改该答卷")

    a_map = {a.id: a for a in sub.answers}
    for item in body.answers:
        a = a_map.get(item.answer_id)
        if not a:
            continue
        a.score = item.score
        a.feedback = item.feedback
        a.is_correct = item.score >= 0.5
        a.graded_by = "teacher"

    total_score = sum((a.score or 0.0) for a in sub.answers)
    n = max(1, len(sub.answers))
    sub.score = round(total_score / n * 100, 1)
    sub.correct_count = sum(1 for a in sub.answers if (a.score or 0.0) >= 0.5)
    await db.commit()

    return await _submission_out(sub, paper, viewer_role="teacher", db=db)
