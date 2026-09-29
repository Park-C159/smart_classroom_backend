"""Learning analytics API — mastery tracking, dashboards."""
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user, get_teacher_or_admin
from app.models import (
    User, Subject, Document, KnowledgePoint, ContentChunk,
    InteractionLog, KPMastery, Discussion, TestQuestion,
    Paper, PaperSubmission, ChunkKpMap,
)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ── Helper ──

async def _get_kp_tree(db: AsyncSession):
    """Fetch all KPs with parent info."""
    result = await db.execute(select(KnowledgePoint).order_by(KnowledgePoint.chapter, KnowledgePoint.sort_order))
    kps = result.scalars().all()
    return [
        {
            "id": kp.id, "title": kp.title, "summary": kp.summary,
            "parent_id": kp.parent_id, "chapter": kp.chapter,
            "level": kp.level, "sort_order": kp.sort_order,
        }
        for kp in kps
    ]


# ── Student: own mastery ──

@router.get("/my-mastery")
async def get_my_mastery(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return the current user's mastery for all KPs (tree structure)."""
    user_id = current_user["user_id"]

    # Get all KPs
    kps = await _get_kp_tree(db)

    # Get user's mastery records
    result = await db.execute(
        select(KPMastery).where(KPMastery.user_id == user_id)
    )
    masteries = {m.kp_id: m for m in result.scalars().all()}

    # Merge
    tree = []
    for kp in kps:
        m = masteries.get(kp["id"])
        tree.append({
            **kp,
            "mastery": round(m.mastery, 3) if m else 0.5,
            "total_questions": m.total_questions if m else 0,
            "last_updated": m.last_updated.isoformat() if m and m.last_updated else None,
        })

    return {"data": tree, "total": len(tree)}


# ── Student: own stats ──

@router.get("/my-stats")
async def get_my_stats(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Return interaction statistics for the current user."""
    user_id = current_user["user_id"]

    # Total questions asked
    total_q = await db.scalar(
        select(func.count(InteractionLog.id)).where(InteractionLog.user_id == user_id)
    )

    # Feedback breakdown
    helpful = await db.scalar(
        select(func.count(InteractionLog.id)).where(
            and_(InteractionLog.user_id == user_id, InteractionLog.feedback == "helpful")
        )
    )
    not_helpful = await db.scalar(
        select(func.count(InteractionLog.id)).where(
            and_(InteractionLog.user_id == user_id, InteractionLog.feedback == "not_helpful")
        )
    )

    # Distinct KPs interacted with
    kp_count = await db.scalar(
        select(func.count(InteractionLog.id)).where(InteractionLog.user_id == user_id)
    )

    # Average mastery across all KPs
    avg_mastery = await db.scalar(
        select(func.avg(KPMastery.mastery)).where(KPMastery.user_id == user_id)
    )

    # Recent interactions
    result = await db.execute(
        select(InteractionLog)
        .where(InteractionLog.user_id == user_id)
        .order_by(InteractionLog.created_at.desc())
        .limit(10)
    )
    recent = result.scalars().all()

    # 作业/测试/考试（新组卷）统计
    total_papers = await db.scalar(
        select(func.count(PaperSubmission.id)).where(PaperSubmission.user_id == user_id)
    )
    graded_papers = await db.scalar(
        select(func.count(PaperSubmission.id)).where(
            and_(PaperSubmission.user_id == user_id, PaperSubmission.status == "submitted")
        )
    )
    avg_paper_score = await db.scalar(
        select(func.avg(PaperSubmission.score)).where(
            and_(PaperSubmission.user_id == user_id, PaperSubmission.status == "submitted")
        )
    )
    mode_rows = await db.execute(
        select(Paper.mode, func.count(PaperSubmission.id))
        .join(Paper, Paper.id == PaperSubmission.paper_id)
        .where(PaperSubmission.user_id == user_id)
        .group_by(Paper.mode)
    )
    papers_by_mode = {mode: cnt for mode, cnt in mode_rows.all()}

    return {
        "total_questions": total_q or 0,
        "helpful_count": helpful or 0,
        "not_helpful_count": not_helpful or 0,
        "helpful_rate": round(helpful / total_q, 2) if total_q else 0,
        "avg_kp_mastery": round(avg_mastery, 3) if avg_mastery else 0.5,
        "total_exams": total_papers or 0,
        "graded_exams": graded_papers or 0,
        "avg_exam_score": round(avg_paper_score, 1) if avg_paper_score else None,
        "total_papers": total_papers or 0,
        "graded_papers": graded_papers or 0,
        "avg_paper_score": round(avg_paper_score, 1) if avg_paper_score else None,
        "papers_by_mode": papers_by_mode,
        "recent_interactions": [
            {
                "id": r.id,
                "question": r.question[:100],
                "feedback": r.feedback,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in recent
        ],
    }


# ── Student: chapter mastery radar（按学科知识树的「章」划分） ──

_CN_DIGITS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}

# 答疑对掌握度的单次权重。题目按难度（1-5）加权，答疑量远多于作业/测试，
# 故答疑权重需显著低于题目权重，避免海量提问淹没少量作业结果。
QA_WEIGHT = 0.2


def _parse_chapter_no(text: str) -> int | None:
    """从「第N章 xxx」中解析章号 N（支持阿拉伯数字与中文数字）。"""
    if not text:
        return None
    m = re.search(r"第\s*([0-9一二三四五六七八九十百]+)\s*章", text)
    if not m:
        return None
    s = m.group(1)
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if s in _CN_DIGITS:
        return _CN_DIGITS[s]
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGITS.get(left, 1)
        ones = _CN_DIGITS.get(right, 0)
        return tens * 10 + ones
    return None


def _chapter_key(title: str) -> str:
    """去掉章标题的「第N章」前缀，得到用于匹配的内容部分。"""
    return re.sub(r"^第\s*[0-9一二三四五六七八九十百]+\s*章\s*", "", title or "").strip()


async def _compute_chapter_radar(db, user_id: int, subject_id: int):
    """按学科知识树的章节计算学生掌握度（0-1，无证据默认 0.5）。

    证据来源：
      1) 作业/测试逐题结果：正确率（0-1）× 难度（1-5）加权；
      2) 答疑反馈：helpful/未反馈=1.0 / not_helpful=0.0，权重 QA_WEIGHT（默认 0.2）。
    题目按节（kp_id）归到其所属章；答疑按 matched_kps 的章标题归章。
    """
    from app.models import Paper, PaperAnswer, PaperQuestion, PaperSubmission, InteractionLog

    DEFAULT_SCORE = 0.5

    chapters = (await db.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.subject_id == subject_id, KnowledgePoint.level == 0)
        .order_by(KnowledgePoint.sort_order, KnowledgePoint.id)
    )).scalars().all()

    if not chapters:
        return {"subject_id": subject_id, "dimensions": [], "overall": DEFAULT_SCORE, "total_answered": 0}

    by_id = {c.id: c for c in chapters}
    by_no = {}
    by_title = {}
    for c in chapters:
        m = re.search(r"(\d+)$", str(c.id))
        if m:
            by_no[int(m.group(1))] = c.id
        key = _chapter_key(c.title)
        if key:
            by_title[key] = c.id

    # 节 -> 章
    sec_rows = (await db.execute(
        select(KnowledgePoint).where(KnowledgePoint.parent_id.in_(list(by_id.keys())))
    )).scalars().all()
    kp_to_chapter = {s.id: s.parent_id for s in sec_rows}

    agg = {cid: {"sum_score": 0.0, "sum_weight": 0.0, "answered": 0, "qa": 0, "qa_helpful": 0, "qa_not_helpful": 0} for cid in by_id}

    # 1) 作业/测试逐题结果
    stmt = (
        select(PaperAnswer.score, PaperAnswer.is_correct, PaperQuestion.kp_id, PaperQuestion.difficulty)
        .join(PaperSubmission, PaperAnswer.submission_id == PaperSubmission.id)
        .join(PaperQuestion, PaperAnswer.paper_question_id == PaperQuestion.id)
        .join(Paper, PaperSubmission.paper_id == Paper.id)
        .where(and_(
            PaperSubmission.user_id == user_id,
            PaperSubmission.status == "submitted",
            Paper.subject_id == subject_id,
        ))
    )
    for score, is_correct, kp_id, difficulty in (await db.execute(stmt)).all():
        cid = kp_to_chapter.get(kp_id)
        if cid not in agg:
            continue
        s = score if score is not None else (1.0 if is_correct else 0.0)
        if s is None:
            continue
        s = max(0.0, min(1.0, s))
        w = float(difficulty or 3)
        a = agg[cid]
        a["sum_score"] += s * w
        a["sum_weight"] += w
        a["answered"] += 1

    # 2) 答疑反馈（helpful/not_helpful）→ 章
    logs = (await db.execute(
        select(InteractionLog.matched_kps, InteractionLog.feedback)
        .where(InteractionLog.user_id == user_id, InteractionLog.matched_kps.isnot(None))
    )).all()
    for matched_kps, feedback in logs:
        if feedback == "not_helpful":
            qa_score = 0.0
        else:
            qa_score = 1.0  # 有帮助 或 未反馈（默认视为有帮助）
        entries = []
        if isinstance(matched_kps, dict):
            entries = matched_kps.get("chapters") or matched_kps.get("kps") or []
        cids = set()
        for e in entries:
            if isinstance(e, dict):
                kp_id = e.get("kp_id")
                cid = kp_to_chapter.get(kp_id) or (kp_id if kp_id in by_id else None)
                if cid:
                    cids.add(cid)
            elif isinstance(e, str):
                no = _parse_chapter_no(e)
                cid = by_no.get(no) if no else None
                if not cid:
                    cid = by_title.get(_chapter_key(e)) or (e.strip() if e.strip() in by_id else None)
                if cid:
                    cids.add(cid)
        for cid in cids:
            a = agg[cid]
            a["sum_score"] += qa_score * QA_WEIGHT
            a["sum_weight"] += QA_WEIGHT
            a["qa"] += 1
            if feedback == "helpful":
                a["qa_helpful"] += 1
            elif feedback == "not_helpful":
                a["qa_not_helpful"] += 1

    # 3) 汇总（无证据默认 0.5）
    total_score = 0.0
    total_weight = 0.0
    total_answered = 0
    dimensions = []
    for c in chapters:
        a = agg[c.id]
        score = round(a["sum_score"] / a["sum_weight"], 3) if a["sum_weight"] else DEFAULT_SCORE
        dimensions.append({
            "id": c.id,
            "name": c.title or c.id,
            "score": score,
            "answered": a["answered"],
            "qa_count": a["qa"],
            "qa_helpful": a["qa_helpful"],
            "qa_not_helpful": a["qa_not_helpful"],
        })
        if a["sum_weight"]:
            total_score += a["sum_score"]
            total_weight += a["sum_weight"]
            total_answered += a["answered"]

    return {
        "subject_id": subject_id,
        "dimensions": dimensions,
        "overall": round(total_score / total_weight, 3) if total_weight else DEFAULT_SCORE,
        "total_answered": total_answered,
    }


@router.get("/chapter-radar")
async def get_chapter_radar(
    subject_id: int | None = None,
    user_id: int | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """章节掌握雷达：维度 = 该学科知识树的各章，学生看自己，教师/管理员可指定 user_id。"""
    target = current_user["user_id"]
    if user_id is not None:
        if current_user["role"] not in ("teacher", "admin"):
            if user_id != target:
                raise HTTPException(403, "只能查看自己的学情分析")
        else:
            target = user_id
    if not subject_id:
        subject_id = await db.scalar(select(Subject.id).order_by(Subject.id).limit(1))
    if not subject_id:
        return {"subject_id": None, "dimensions": [], "overall": 0.5, "total_answered": 0}
    return await _compute_chapter_radar(db, target, subject_id)


async def _compute_section_radar(db, user_id: int, subject_id: int, chapter_id: str):
    """某章下各节的掌握度雷达（与章雷达同口径：作业测试正确率×难度 + 答疑反馈，无证据默认 0.5）。"""
    from app.models import Paper, PaperAnswer, PaperQuestion, PaperSubmission, InteractionLog

    DEFAULT_SCORE = 0.5
    sections = (await db.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.subject_id == subject_id, KnowledgePoint.parent_id == chapter_id)
        .order_by(KnowledgePoint.sort_order, KnowledgePoint.id)
    )).scalars().all()
    if not sections:
        return {"chapter_id": chapter_id, "dimensions": [], "overall": DEFAULT_SCORE, "total_answered": 0}

    sec_ids = [s.id for s in sections]
    agg = {s.id: {"sum_score": 0.0, "sum_weight": 0.0, "answered": 0, "qa": 0, "qa_helpful": 0, "qa_not_helpful": 0} for s in sections}

    # 1) 作业/测试逐题结果
    stmt = (
        select(PaperAnswer.score, PaperAnswer.is_correct, PaperQuestion.kp_id, PaperQuestion.difficulty)
        .join(PaperSubmission, PaperAnswer.submission_id == PaperSubmission.id)
        .join(PaperQuestion, PaperAnswer.paper_question_id == PaperQuestion.id)
        .join(Paper, PaperSubmission.paper_id == Paper.id)
        .where(and_(
            PaperSubmission.user_id == user_id,
            PaperSubmission.status == "submitted",
            Paper.subject_id == subject_id,
            PaperQuestion.kp_id.in_(sec_ids),
        ))
    )
    for score, is_correct, kp_id, difficulty in (await db.execute(stmt)).all():
        a = agg.get(kp_id)
        if a is None:
            continue
        s = score if score is not None else (1.0 if is_correct else 0.0)
        if s is None:
            continue
        s = max(0.0, min(1.0, s))
        w = float(difficulty or 3)
        a["sum_score"] += s * w
        a["sum_weight"] += w
        a["answered"] += 1

    # 2) 答疑反馈（helpful/未反馈=1.0，not_helpful=0.0）→ 节
    logs = (await db.execute(
        select(InteractionLog.matched_kps, InteractionLog.feedback)
        .where(InteractionLog.user_id == user_id, InteractionLog.matched_kps.isnot(None))
    )).all()
    for matched_kps, feedback in logs:
        qa_score = 0.0 if feedback == "not_helpful" else 1.0
        entries = matched_kps.get("sections") or [] if isinstance(matched_kps, dict) else []
        for e in entries:
            sid = e.get("kp_id") if isinstance(e, dict) else e
            a = agg.get(sid)
            if a is None:
                continue
            a["sum_score"] += qa_score * QA_WEIGHT
            a["sum_weight"] += QA_WEIGHT
            a["qa"] += 1
            if feedback == "helpful":
                a["qa_helpful"] += 1
            elif feedback == "not_helpful":
                a["qa_not_helpful"] += 1

    # 3) 汇总（无证据默认 0.5）
    total_score = 0.0
    total_weight = 0.0
    total_answered = 0
    dimensions = []
    for s in sections:
        a = agg[s.id]
        score = round(a["sum_score"] / a["sum_weight"], 3) if a["sum_weight"] else DEFAULT_SCORE
        dimensions.append({
            "id": s.id, "name": s.title or s.id, "score": score,
            "answered": a["answered"], "qa_count": a["qa"],
            "qa_helpful": a["qa_helpful"], "qa_not_helpful": a["qa_not_helpful"],
        })
        if a["sum_weight"]:
            total_score += a["sum_score"]
            total_weight += a["sum_weight"]
            total_answered += a["answered"]

    return {
        "chapter_id": chapter_id,
        "dimensions": dimensions,
        "overall": round(total_score / total_weight, 3) if total_weight else DEFAULT_SCORE,
        "total_answered": total_answered,
    }


@router.get("/section-radar")
async def get_section_radar(
    subject_id: int,
    chapter_id: str,
    user_id: int | None = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """某章各节的掌握度雷达（点击章名下钻）。"""
    target = current_user["user_id"]
    if user_id is not None:
        if current_user["role"] not in ("teacher", "admin"):
            if user_id != target:
                raise HTTPException(403, "只能查看自己的学情分析")
        else:
            target = user_id
    return await _compute_section_radar(db, target, subject_id, chapter_id)


# ── Teacher: class dashboard ──

@router.get("/class/{class_name}")
async def get_class_analytics(
    class_name: str,
    current_user: dict = Depends(get_teacher_or_admin),
    db: AsyncSession = Depends(get_db),
):
    """Return aggregated analytics for a class (teacher/admin only)."""
    # Get students in this class
    result = await db.execute(
        select(User).where(
            and_(User.class_name == class_name, User.role == "student", User.is_active == True)
        )
    )
    students = result.scalars().all()

    if not students:
        return {"class_name": class_name, "student_count": 0, "students": [], "kp_mastery_avg": []}

    student_ids = [s.id for s in students]

    # Aggregate KP mastery
    result = await db.execute(
        select(
            KPMastery.kp_id,
            func.avg(KPMastery.mastery).label("avg_mastery"),
            func.count(KPMastery.user_id).label("student_count"),
        )
        .where(KPMastery.user_id.in_(student_ids))
        .group_by(KPMastery.kp_id)
    )
    kp_agg = {row.kp_id: {"avg_mastery": round(row.avg_mastery, 3), "student_count": row.student_count}
              for row in result.all()}

    # Per-student summary
    student_summaries = []
    for s in students:
        q_count = await db.scalar(
            select(func.count(InteractionLog.id)).where(InteractionLog.user_id == s.id)
        )
        avg_m = await db.scalar(
            select(func.avg(KPMastery.mastery)).where(KPMastery.user_id == s.id)
        )
        student_summaries.append({
            "user_id": s.id,
            "username": s.username,
            "real_name": s.real_name,
            "total_questions": q_count or 0,
            "avg_mastery": round(avg_m, 3) if avg_m else 0.5,
        })

    # Get KP titles
    kps = await _get_kp_tree(db)
    kp_map = {kp["id"]: kp["title"] for kp in kps}

    return {
        "class_name": class_name,
        "student_count": len(students),
        "students": student_summaries,
        "kp_mastery_avg": [
            {"kp_id": k, "kp_title": kp_map.get(k, k), **v}
            for k, v in sorted(kp_agg.items(), key=lambda x: x[1]["avg_mastery"])
        ],
    }


# ── Admin: overall dashboard ──

@router.get("/dashboard")
async def get_dashboard(
    current_user: dict = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    """Return system-wide analytics dashboard (admin only)."""
    # Counts
    total_users = await db.scalar(select(func.count(User.id)))
    active_users = await db.scalar(
        select(func.count(User.id)).where(User.is_active == True)
    )
    student_count = await db.scalar(
        select(func.count(User.id)).where(User.role == "student")
    )
    teacher_count = await db.scalar(
        select(func.count(User.id)).where(User.role == "teacher")
    )
    admin_count = await db.scalar(
        select(func.count(User.id)).where(User.role == "admin")
    )

    total_documents = await db.scalar(select(func.count(Document.id)))
    total_exercises = await db.scalar(select(func.count(TestQuestion.id)))
    total_discussions = await db.scalar(select(func.count(Discussion.id)))
    total_exams = await db.scalar(select(func.count(Paper.id)))
    total_interactions = await db.scalar(select(func.count(InteractionLog.id)))
    total_kps = await db.scalar(select(func.count(KnowledgePoint.id)))
    total_chunks = await db.scalar(select(func.count(ContentChunk.id)))

    # Interaction trend (last 7 days)
    from datetime import datetime, timezone, timedelta
    seven_days_ago = datetime.now(timezone.utc) - timedelta(days=7)
    result = await db.execute(
        select(func.date(InteractionLog.created_at).label("day"), func.count(InteractionLog.id))
        .where(InteractionLog.created_at >= seven_days_ago)
        .group_by(func.date(InteractionLog.created_at))
        .order_by("day")
    )
    trend = [{"date": str(row.day), "count": row.count} for row in result.all()]

    # Top KPs by interaction
    result = await db.execute(
        select(KnowledgePoint.id, KnowledgePoint.title, func.count(InteractionLog.id).label("cnt"))
        .join(InteractionLog, InteractionLog.matched_kps.isnot(None))
        .where(InteractionLog.matched_kps != None)
        .group_by(KnowledgePoint.id, KnowledgePoint.title)
        .order_by(func.count(InteractionLog.id).desc())
        .limit(10)
    )
    top_kps = [{"kp_id": row.id, "kp_title": row.title, "interaction_count": row.cnt}
               for row in result.all()]

    # Document status breakdown
    pending_docs = await db.scalar(
        select(func.count(Document.id)).where(Document.status == "pending")
    )
    processing_docs = await db.scalar(
        select(func.count(Document.id)).where(Document.status == "processing")
    )
    completed_docs = await db.scalar(
        select(func.count(Document.id)).where(Document.status == "completed")
    )

    return {
        "users": {
            "total": total_users or 0,
            "active": active_users or 0,
            "students": student_count or 0,
            "teachers": teacher_count or 0,
            "admins": admin_count or 0,
        },
        "content": {
            "documents": total_documents or 0,
            "pending": pending_docs or 0,
            "processing": processing_docs or 0,
            "completed": completed_docs or 0,
            "exercises": total_exercises or 0,
            "knowledge_points": total_kps or 0,
            "total_chunks": total_chunks or 0,
        },
        "activity": {
            "total_interactions": total_interactions or 0,
            "total_discussions": total_discussions or 0,
            "total_exams": total_exams or 0,
            "trend_7d": trend,
        },
        "top_kps": top_kps,
    }


# ── KP-level statistics ──

@router.get("/kp/{kp_id}/stats")
async def get_kp_stats(
    kp_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Return statistics for a specific knowledge point."""
    # KP info
    result = await db.execute(select(KnowledgePoint).where(KnowledgePoint.id == kp_id))
    kp = result.scalar_one_or_none()
    if not kp:
        return {"error": "Knowledge point not found", "kp_id": kp_id}

    # Content chunks count（分块按 chunk_kp_map 归属章/节）
    chunk_count = await db.scalar(
        select(func.count(ChunkKpMap.chunk_id)).where(
            or_(ChunkKpMap.chapter_id == kp_id, ChunkKpMap.section_id == kp_id)
        )
    )

    # Exercise count（用独立试题库 TestQuestion 统计）
    ex_count = await db.scalar(
        select(func.count(TestQuestion.id)).where(TestQuestion.kp_id == kp_id)
    )

    # Average mastery across users
    avg_m = await db.scalar(
        select(func.avg(KPMastery.mastery)).where(KPMastery.kp_id == kp_id)
    )
    mastery_count = await db.scalar(
        select(func.count(KPMastery.user_id)).where(KPMastery.kp_id == kp_id)
    )

    # Exercise count by difficulty
    result = await db.execute(
        select(TestQuestion.difficulty, func.count(TestQuestion.id))
        .where(TestQuestion.kp_id == kp_id)
        .group_by(TestQuestion.difficulty)
    )
    diff_dist = {f"level_{row.difficulty}": row.count for row in result.all()}

    return {
        "kp_id": kp.id,
        "kp_title": kp.title,
        "chapter": kp.chapter,
        "content_chunks": chunk_count or 0,
        "exercises": ex_count or 0,
        "avg_mastery": round(avg_m, 3) if avg_m else None,
        "users_tracked": mastery_count or 0,
        "difficulty_distribution": diff_dist,
    }
