"""Feedback API — user feedback submission and admin review."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user
from app.models import Feedback
from app.schemas.schemas import FeedbackCreate, FeedbackOut, PaginatedResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/feedbacks", tags=["feedback"])


@router.post("")
async def create_feedback(
    data: FeedbackCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Submit feedback (any authenticated user)."""
    fb = Feedback(
        user_id=int(current_user["user_id"]),
        category=data.category,
        content=data.content,
        attachment=data.attachment,
    )
    db.add(fb)
    await db.commit()
    await db.refresh(fb)
    return {"message": "反馈已提交，感谢你的建议！", "id": fb.id}


@router.get("")
async def list_feedbacks(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """List feedbacks (admin only). Supports category filter and pagination."""
    from sqlalchemy.orm import selectinload

    query = select(Feedback).options(selectinload(Feedback.user))

    if category:
        query = query.where(Feedback.category == category)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Fetch page
    query = query.order_by(Feedback.created_at.desc())
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    feedbacks = result.scalars().all()

    items = []
    for f in feedbacks:
        items.append({
            "id": f.id,
            "user_id": f.user_id,
            "author_name": f.user.real_name or f.user.username if f.user else "",
            "category": f.category,
            "content": f.content,
            "attachment": f.attachment,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        })

    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        total_pages=max(1, (total + page_size - 1) // page_size),
    )
