"""Exam generation, submission, and grading API."""
import random
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, and_, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user
from app.models import (
    User, Subject, KnowledgePoint, QuestionBank, KPMastery,
    Exam, ExamQuestion,
)

router = APIRouter(prefix="/api/exam", tags=["exam"])


# ── Pydantic models ──

class GenerateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    subject_id: int | None = None
    kp_ids: list[str] = []                # target KPs; if empty, auto-select weakest
    chapters: list[str] = []              # target chapters（章节筛选，优先于 kp_ids）
    question_count: int = Field(default=10, ge=1, le=50)
    difficulty_min: int = Field(default=1, ge=1, le=5)
    difficulty_max: int = Field(default=5, ge=1, le=5)


class AnswerInput(BaseModel):
    question_id: int
    user_answer: str


class SubmitRequest(BaseModel):
    answers: list[AnswerInput] = Field(..., min_length=1)


# ── Helpers ──

def _exam_to_out(exam: Exam) -> dict:
    return {
        "id": exam.id,
        "user_id": exam.user_id,
        "title": exam.title,
        "subject_id": exam.subject_id,
        "kp_ids": exam.kp_ids,
        "total_questions": exam.total_questions,
        "correct_count": exam.correct_count,
        "score": exam.score,
        "status": exam.status,
        "created_at": exam.created_at.isoformat() if exam.created_at else None,
    }


_ANSWER_MARKERS = [
    r'\n\s*解\s*[:：]?',
    r'\n\s*答\s*[:：]?',
    r'\n\s*答案\s*[:：]?',
    r'\n\s*略解\s*[:：]?',
]


def _split_question_answer(text: str) -> tuple:
    """题库里题目与答案混在一段（answer_text 为空）时，尽量拆出答案。

    教材例题格式多为「例X 求/证明… 。\\n\\n 解答…」：题目以句号结尾后跟空行，
    之后是解答。这里在组卷时拆分，避免试卷直接露出答案。
    """
    if not text:
        return text, None
    # 显式「解/答/答案」标记
    for pat in _ANSWER_MARKERS:
        m = re.search(pat, text, re.MULTILINE)
        if m and m.start() > 10:
            return text[:m.start()].strip(), text[m.start():].strip()
    # 题目句号 + 空行 → 之后为解答
    m = re.search(
        r'(?:求|证明|计算|判断|化简|求证|表为|解|试证|讨论).*?[。．\.]\n\s*\n',
        text, re.DOTALL,
    )
    if m:
        return text[:m.end()].strip(), text[m.end():].strip()
    return text, None


# ── Generate ──

@router.post("/generate")
async def generate_exam(
    body: GenerateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Generate an exam by picking exercises for target KPs."""
    user_id = current_user["user_id"]

    # 章节优先：直接按章节筛选；否则用 kp_ids（自动选薄弱点）映射章节
    chapters = list(body.chapters or [])
    kp_ids = list(body.kp_ids or [])

    if not chapters:
        if not kp_ids:
            # Auto-select weakest KPs (lowest mastery)
            result = await db.execute(
                select(KPMastery.kp_id)
                .where(KPMastery.user_id == user_id)
                .order_by(KPMastery.mastery.asc())
                .limit(5)
            )
            kp_ids = [row[0] for row in result.all()]

        if not kp_ids:
            # No mastery data yet — pick any KPs with exercises
            result = await db.execute(select(KnowledgePoint.id).limit(5))
            kp_ids = [row[0] for row in result.all()]

        if not kp_ids:
            raise HTTPException(status_code=400, detail="No knowledge points available. Upload and parse a textbook first.")

        # 检索题库 QuestionBank 用 chapter（章级）关联，kp_id 当前未填充 → 先把 kp_ids 映射成章节名
        ch_result = await db.execute(
            select(KnowledgePoint.chapter).where(
                and_(KnowledgePoint.id.in_(kp_ids), KnowledgePoint.chapter.is_not(None))
            )
        )
        chapters = list({row[0] for row in ch_result.all()})

    # Fetch exercises matching the chapters and difficulty range（组卷用检索题库 QuestionBank）
    # 只用「习题」（question_type != example），不用教材「例题」（例X 含完整解答，不好拆）
    conditions = [
        QuestionBank.difficulty >= body.difficulty_min,
        QuestionBank.difficulty <= body.difficulty_max,
        QuestionBank.question_type != "example",
    ]
    if chapters:
        conditions.append(QuestionBank.chapter.in_(chapters))
    result = await db.execute(select(QuestionBank).where(and_(*conditions)))
    all_exercises = result.scalars().all()
    exercises = random.sample(list(all_exercises), min(len(all_exercises), body.question_count)) if all_exercises else []

    if not exercises:
        raise HTTPException(
            status_code=400,
            detail="No exercises found for the selected knowledge points and difficulty range."
        )

    # Create exam
    exam = Exam(
        user_id=user_id,
        title=body.title,
        subject_id=body.subject_id,
        kp_ids=kp_ids,
        total_questions=len(exercises),
        status="draft",
    )
    db.add(exam)
    await db.flush()

    # Create exam questions
    for i, ex in enumerate(exercises):
        q_text, a_text = ex.question_text, ex.answer_text
        if not a_text:
            # 题库题目与答案混在一起时，组卷时拆开，试卷只显示题目
            q_text, a_text = _split_question_answer(q_text)
        eq = ExamQuestion(
            exam_id=exam.id,
            exercise_id=ex.id,
            question_text=q_text,
            answer_text=a_text,
            question_type=ex.question_type,
            difficulty=ex.difficulty,
            sort_order=i,
        )
        db.add(eq)

    await db.flush()

    # Fetch with questions
    result = await db.execute(
        select(Exam)
        .where(Exam.id == exam.id)
        .options(selectinload(Exam.questions))
    )
    exam = result.scalar_one()

    return {
        "exam": _exam_to_out(exam),
        "questions": [
            {
                "id": q.id,
                "exercise_id": q.exercise_id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "difficulty": q.difficulty,
                "sort_order": q.sort_order,
                # Don't expose answer_text during draft
            }
            for q in exam.questions
        ],
    }


# ── List my exams ──

@router.get("/my-exams")
async def list_my_exams(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List the current user's exams."""
    user_id = current_user["user_id"]
    offset = (page - 1) * page_size

    total = await db.scalar(
        select(func.count(Exam.id)).where(Exam.user_id == user_id)
    )
    result = await db.execute(
        select(Exam)
        .where(Exam.user_id == user_id)
        .order_by(Exam.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    exams = result.scalars().all()

    return {
        "data": [_exam_to_out(e) for e in exams],
        "total": total or 0,
        "page": page,
        "page_size": page_size,
    }


# ── List all exams (admin/teacher) ──

@router.get("/")
async def list_all_exams(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    user_id: int | None = None,
    status: str | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all exams (admin can filter by user and status)."""
    role = current_user["role"]
    user_uid = current_user["user_id"]

    # Students can only see their own
    if role == "student":
        user_id = user_uid

    conditions = []
    if user_id:
        conditions.append(Exam.user_id == user_id)
    if status:
        conditions.append(Exam.status == status)

    offset = (page - 1) * page_size
    base = select(Exam)
    if conditions:
        base = base.where(and_(*conditions))

    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    result = await db.execute(
        base.order_by(Exam.created_at.desc())
        .offset(offset)
        .limit(page_size)
    )
    exams = result.scalars().all()

    return {
        "data": [_exam_to_out(e) for e in exams],
        "total": total or 0,
        "page": page,
        "page_size": page_size,
    }


# ── Get exam detail ──

@router.get("/{exam_id}")
async def get_exam(
    exam_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get exam with all questions."""
    result = await db.execute(
        select(Exam)
        .where(Exam.id == exam_id)
        .options(selectinload(Exam.questions))
    )
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    # Access control: students can only see their own exams
    if current_user["role"] == "student" and exam.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    questions_out = []
    for q in sorted(exam.questions, key=lambda x: x.sort_order):
        qd = {
            "id": q.id,
            "exercise_id": q.exercise_id,
            "question_text": q.question_text,
            "question_type": q.question_type,
            "difficulty": q.difficulty,
            "sort_order": q.sort_order,
        }
        # Show answer only if graded
        if exam.status == "graded":
            qd["answer_text"] = q.answer_text
            qd["user_answer"] = q.user_answer
            qd["is_correct"] = q.is_correct
        # Show answer only to owner after submitted
        elif exam.status == "submitted" and current_user["user_id"] == exam.user_id:
            qd["user_answer"] = q.user_answer
        exam_out = qd

    return {
        "exam": _exam_to_out(exam),
        "questions": [
            {
                "id": q.id,
                "exercise_id": q.exercise_id,
                "question_text": q.question_text,
                "question_type": q.question_type,
                "difficulty": q.difficulty,
                "sort_order": q.sort_order,
                **(
                    {"answer_text": q.answer_text, "user_answer": q.user_answer, "is_correct": q.is_correct}
                    if exam.status == "graded" else
                    {"user_answer": q.user_answer}
                    if exam.status == "submitted" and current_user["user_id"] == exam.user_id else
                    {}
                ),
            }
            for q in sorted(exam.questions, key=lambda x: x.sort_order)
        ],
    }


# ── Submit answers ──

@router.post("/{exam_id}/submit")
async def submit_exam(
    exam_id: int,
    body: SubmitRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit answers for grading."""
    result = await db.execute(
        select(Exam)
        .where(Exam.id == exam_id)
        .options(selectinload(Exam.questions))
    )
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    if exam.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    if exam.status == "graded":
        raise HTTPException(status_code=400, detail="Exam already graded")

    # Build question lookup
    q_map = {q.id: q for q in exam.questions}

    # Record answers and grade
    correct = 0
    for ans in body.answers:
        q = q_map.get(ans.question_id)
        if not q:
            continue
        q.user_answer = ans.user_answer
        # Simple answer comparison (case-insensitive, strip whitespace)
        if q.answer_text:
            q.is_correct = (
                ans.user_answer.strip().lower() == q.answer_text.strip().lower()
            )
        if q.is_correct:
            correct += 1

    exam.status = "graded"
    exam.correct_count = correct
    exam.score = round(correct / exam.total_questions * 100, 1) if exam.total_questions else 0

    # 更新知识点掌握度（学情分析）：按试卷命中的知识点，用整体正确率做指数滑动平均
    correct_rate = correct / exam.total_questions if exam.total_questions else 0.5
    for kp_id in (exam.kp_ids or []):
        mres = await db.execute(
            select(KPMastery).where(
                KPMastery.user_id == exam.user_id,
                KPMastery.kp_id == kp_id,
            )
        )
        m = mres.scalar_one_or_none()
        if m:
            m.mastery = round(max(0.1, min(1.0, m.mastery * 0.7 + correct_rate * 0.3)), 3)
            m.total_questions += 1
        else:
            db.add(KPMastery(
                user_id=exam.user_id,
                kp_id=kp_id,
                mastery=round(max(0.1, min(1.0, 0.5 * 0.7 + correct_rate * 0.3)), 3),
                total_questions=1,
            ))

    await db.flush()

    # Return graded result
    return {
        "exam": _exam_to_out(exam),
        "questions": [
            {
                "id": q.id,
                "exercise_id": q.exercise_id,
                "question_text": q.question_text,
                "answer_text": q.answer_text,
                "question_type": q.question_type,
                "difficulty": q.difficulty,
                "user_answer": q.user_answer,
                "is_correct": q.is_correct,
                "sort_order": q.sort_order,
            }
            for q in sorted(exam.questions, key=lambda x: x.sort_order)
        ],
    }


# ── Delete exam ──

@router.delete("/{exam_id}")
async def delete_exam(
    exam_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Delete an exam (owner or admin)."""
    result = await db.execute(select(Exam).where(Exam.id == exam_id))
    exam = result.scalar_one_or_none()
    if not exam:
        raise HTTPException(status_code=404, detail="Exam not found")

    if current_user["role"] != "admin" and exam.user_id != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Access denied")

    # SQLite 未启用外键级联（ON DELETE CASCADE 不生效），先手动删关联题目再删卷子
    await db.execute(delete(ExamQuestion).where(ExamQuestion.exam_id == exam_id))
    await db.delete(exam)
    return {"message": "Exam deleted successfully"}
