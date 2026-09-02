"""Discussion forum API — posts, replies, likes."""
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, func, delete, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user, get_teacher_or_admin
from app.models import (
    Discussion, DiscussionReply, DiscussionLike, ReplyLike, User,
)
from app.schemas.schemas import (
    DiscussionCreate, DiscussionOut, DiscussionReplyCreate, DiscussionReplyOut,
    PaginatedResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/discussion", tags=["discussion"])


# ── Helpers ──

def _discussion_to_out(d: Discussion, current_user_id: int, is_liked: bool = False) -> dict:
    """Convert Discussion ORM object to output dict."""
    return {
        "id": d.id,
        "title": d.title,
        "content": d.content,
        "user_id": d.user_id,
        "author_name": d.author.real_name or d.author.username if d.author else "",
        "author_avatar": d.author.avatar_url if d.author else None,
        "subject_id": d.subject_id,
        "kp_id": d.kp_id,
        "qa_refs": d.qa_refs if d.qa_refs else None,
        "is_pinned": d.is_pinned,
        "view_count": d.view_count,
        "like_count": d.like_count,
        "reply_count": d.reply_count,
        "is_liked": is_liked,
        "created_at": d.created_at.isoformat() if d.created_at else None,
        "updated_at": d.updated_at.isoformat() if d.updated_at else None,
    }


def _reply_to_out(r: DiscussionReply, current_user_id: int, is_liked: bool = False) -> dict:
    """Convert DiscussionReply ORM object to output dict."""
    return {
        "id": r.id,
        "discussion_id": r.discussion_id,
        "user_id": r.user_id,
        "author_name": r.user.real_name or r.user.username if r.user else "",
        "author_avatar": r.user.avatar_url if r.user else None,
        "content": r.content,
        "parent_id": r.parent_id,
        "like_count": r.like_count,
        "is_liked": is_liked,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


# ── Discussion CRUD ──

@router.get("")
async def list_discussions(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    subject_id: Optional[int] = None,
    kp_id: Optional[str] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List discussions with pagination and filters."""
    query = select(Discussion).options(selectinload(Discussion.author))

    if subject_id:
        query = query.where(Discussion.subject_id == subject_id)
    if kp_id:
        query = query.where(Discussion.kp_id == kp_id)
    if search:
        query = query.where(
            (Discussion.title.ilike(f"%{search}%"))
            | (Discussion.content.ilike(f"%{search}%"))
        )

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Fetch page (pinned first, then by created_at desc)
    query = query.order_by(Discussion.is_pinned.desc(), Discussion.created_at.desc())
    offset = (page - 1) * page_size
    query = query.offset(offset).limit(page_size)
    result = await db.execute(query)
    discussions = result.scalars().all()

    # Check likes for current user
    user_id = int(current_user["user_id"])
    liked_ids = set()
    if discussions:
        disc_ids = [d.id for d in discussions]
        likes_result = await db.execute(
            select(DiscussionLike.discussion_id).where(
                and_(DiscussionLike.user_id == user_id, DiscussionLike.discussion_id.in_(disc_ids))
            )
        )
        liked_ids = {r[0] for r in likes_result.all()}

    items = [_discussion_to_out(d, user_id, d.id in liked_ids) for d in discussions]

    return PaginatedResponse(
        items=items, total=total, page=page, page_size=page_size,
        total_pages=max(1, (total + page_size - 1) // page_size),
    )


@router.post("")
async def create_discussion(
    data: DiscussionCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Create a new discussion post."""
    disc = Discussion(
        title=data.title,
        content=data.content,
        user_id=int(current_user["user_id"]),
        subject_id=data.subject_id,
        kp_id=data.kp_id,
        qa_refs=data.qa_refs if data.qa_refs else None,
    )
    db.add(disc)
    await db.commit()
    await db.refresh(disc)

    # Reload with author
    result = await db.execute(
        select(Discussion).options(selectinload(Discussion.author)).where(Discussion.id == disc.id)
    )
    disc = result.scalar_one()
    return _discussion_to_out(disc, int(current_user["user_id"]), False)


@router.get("/{discussion_id}")
async def get_discussion(
    discussion_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get discussion with replies."""
    result = await db.execute(
        select(Discussion).options(selectinload(Discussion.author)).where(Discussion.id == discussion_id)
    )
    disc = result.scalar_one_or_none()
    if not disc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "讨论不存在或已删除")

    user_id = int(current_user["user_id"])

    # Check if current user liked
    like_result = await db.execute(
        select(DiscussionLike).where(
            and_(DiscussionLike.user_id == user_id, DiscussionLike.discussion_id == discussion_id)
        )
    )
    is_liked = like_result.scalar_one_or_none() is not None

    # Load replies with authors
    replies_result = await db.execute(
        select(DiscussionReply)
        .options(selectinload(DiscussionReply.user))
        .where(DiscussionReply.discussion_id == discussion_id)
        .order_by(DiscussionReply.created_at.asc())
    )
    replies = replies_result.scalars().all()

    # Check reply likes
    reply_ids = [r.id for r in replies]
    reply_likes = set()
    if reply_ids:
        rl_result = await db.execute(
            select(ReplyLike.reply_id).where(
                and_(ReplyLike.user_id == user_id, ReplyLike.reply_id.in_(reply_ids))
            )
        )
        reply_likes = {r[0] for r in rl_result.all()}

    # Build output BEFORE committing view count
    disc_out = _discussion_to_out(disc, user_id, is_liked)
    replies_out = [_reply_to_out(r, user_id, r.id in reply_likes) for r in replies]

    # Increment view count after building output
    disc.view_count += 1
    await db.commit()

    return {"discussion": disc_out, "replies": replies_out}


@router.delete("/{discussion_id}")
async def delete_discussion(
    discussion_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Delete a discussion (owner or admin only)."""
    result = await db.execute(select(Discussion).where(Discussion.id == discussion_id))
    disc = result.scalar_one_or_none()
    if not disc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "讨论不存在")

    user_id = int(current_user["user_id"])
    is_admin = current_user["role"] == "admin"
    if disc.user_id != user_id and not is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权删除此讨论")

    await db.delete(disc)
    await db.commit()
    return {"message": "讨论已删除"}


@router.post("/{discussion_id}/pin")
async def toggle_pin(
    discussion_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Toggle pin status (teacher/admin only)."""
    result = await db.execute(select(Discussion).where(Discussion.id == discussion_id))
    disc = result.scalar_one_or_none()
    if not disc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "讨论不存在")

    disc.is_pinned = not disc.is_pinned
    disc.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return {"is_pinned": disc.is_pinned, "message": "已置顶" if disc.is_pinned else "已取消置顶"}


# ── Likes ──

@router.post("/{discussion_id}/like")
async def toggle_like(
    discussion_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Toggle like on a discussion."""
    user_id = int(current_user["user_id"])

    # Check discussion exists
    disc_result = await db.execute(select(Discussion).where(Discussion.id == discussion_id))
    disc = disc_result.scalar_one_or_none()
    if not disc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "讨论不存在")

    # Check existing like
    existing = await db.execute(
        select(DiscussionLike).where(
            and_(DiscussionLike.user_id == user_id, DiscussionLike.discussion_id == discussion_id)
        )
    )
    like = existing.scalar_one_or_none()

    if like:
        await db.delete(like)
        disc.like_count = max(0, disc.like_count - 1)
        await db.commit()
        return {"liked": False, "like_count": disc.like_count, "message": "已取消点赞"}
    else:
        db.add(DiscussionLike(user_id=user_id, discussion_id=discussion_id))
        disc.like_count += 1
        await db.commit()
        return {"liked": True, "like_count": disc.like_count, "message": "点赞成功"}


@router.post("/reply/{reply_id}/like")
async def toggle_reply_like(
    reply_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Toggle like on a reply."""
    user_id = int(current_user["user_id"])

    # Check reply exists
    reply_result = await db.execute(select(DiscussionReply).where(DiscussionReply.id == reply_id))
    reply = reply_result.scalar_one_or_none()
    if not reply:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "回复不存在")

    existing = await db.execute(
        select(ReplyLike).where(
            and_(ReplyLike.user_id == user_id, ReplyLike.reply_id == reply_id)
        )
    )
    like = existing.scalar_one_or_none()

    if like:
        await db.delete(like)
        reply.like_count = max(0, reply.like_count - 1)
        await db.commit()
        return {"liked": False, "like_count": reply.like_count, "message": "已取消点赞"}
    else:
        db.add(ReplyLike(user_id=user_id, reply_id=reply_id))
        reply.like_count += 1
        await db.commit()
        return {"liked": True, "like_count": reply.like_count, "message": "点赞成功"}


# ── Replies ──

@router.post("/{discussion_id}/reply")
async def create_reply(
    discussion_id: int,
    data: DiscussionReplyCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Add a reply to a discussion."""
    # Check discussion exists
    disc_result = await db.execute(select(Discussion).where(Discussion.id == discussion_id))
    disc = disc_result.scalar_one_or_none()
    if not disc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "讨论不存在")

    reply = DiscussionReply(
        discussion_id=discussion_id,
        user_id=int(current_user["user_id"]),
        content=data.content,
        parent_id=data.parent_id,
    )
    db.add(reply)

    # Update reply count on discussion
    disc.reply_count += 1
    disc.updated_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(reply)

    # Reload with user
    result = await db.execute(
        select(DiscussionReply).options(selectinload(DiscussionReply.user)).where(DiscussionReply.id == reply.id)
    )
    reply = result.scalar_one()
    return _reply_to_out(reply, int(current_user["user_id"]), False)


@router.delete("/reply/{reply_id}")
async def delete_reply(
    reply_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Delete a reply (owner or admin only)."""
    result = await db.execute(select(DiscussionReply).where(DiscussionReply.id == reply_id))
    reply = result.scalar_one_or_none()
    if not reply:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "回复不存在")

    user_id = int(current_user["user_id"])
    is_admin = current_user["role"] == "admin"
    if reply.user_id != user_id and not is_admin:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "无权删除此回复")

    # Update reply count on parent discussion
    disc_result = await db.execute(select(Discussion).where(Discussion.id == reply.discussion_id))
    disc = disc_result.scalar_one_or_none()
    if disc:
        disc.reply_count = max(0, disc.reply_count - 1)

    await db.delete(reply)
    await db.commit()
    return {"message": "回复已删除"}
