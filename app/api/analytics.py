"""Learning analytics API — mastery tracking, dashboards."""
from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user, get_teacher_or_admin
from app.models import (
    User, Subject, Document, KnowledgePoint, ContentChunk,
    InteractionLog, KPMastery, Discussion, Exam, QuestionBank,
    Paper, PaperSubmission,
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

    # Exam stats
    total_exams = await db.scalar(
        select(func.count(Exam.id)).where(Exam.user_id == user_id)
    )
    graded_exams = await db.scalar(
        select(func.count(Exam.id)).where(
            and_(Exam.user_id == user_id, Exam.status == "graded")
        )
    )
    avg_score = await db.scalar(
        select(func.avg(Exam.score)).where(
            and_(Exam.user_id == user_id, Exam.status == "graded")
        )
    )

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
        "total_exams": total_exams or 0,
        "graded_exams": graded_exams or 0,
        "avg_exam_score": round(avg_score, 1) if avg_score else None,
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
    total_exercises = await db.scalar(select(func.count(QuestionBank.id)))
    total_discussions = await db.scalar(select(func.count(Discussion.id)))
    total_exams = await db.scalar(select(func.count(Exam.id)))
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

    # Content chunks count
    chunk_count = await db.scalar(
        select(func.count(ContentChunk.id)).where(ContentChunk.kp_id == kp_id)
    )

    # Exercise count（用检索题库 QuestionBank 统计）
    ex_count = await db.scalar(
        select(func.count(QuestionBank.id)).where(QuestionBank.kp_id == kp_id)
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
        select(QuestionBank.difficulty, func.count(QuestionBank.id))
        .where(QuestionBank.kp_id == kp_id)
        .group_by(QuestionBank.difficulty)
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
