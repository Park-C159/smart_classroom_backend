"""私信 API — 用户间一对一消息（学生 ↔ 教师）。"""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select, func, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Message, User, UserSubject, Subject

router = APIRouter(prefix="/api/messages", tags=["私信"])


class SendRequest(BaseModel):
    recipient_id: int
    content: str = Field(..., min_length=1, max_length=5000)


def _user_brief(u: User | None) -> dict:
    return {
        "id": u.id,
        "username": u.username,
        "real_name": u.real_name,
        "role": u.role,
    } if u else {"id": None, "username": "未知用户", "real_name": "", "role": ""}


async def _get_user(db: AsyncSession, user_id: int) -> User | None:
    return (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()


async def _subject_ids(db: AsyncSession, user_id: int) -> set[int]:
    res = await db.execute(select(UserSubject.subject_id).where(UserSubject.user_id == user_id))
    return {r[0] for r in res.all()}


async def _can_message(db: AsyncSession, me: User, other: User) -> bool:
    """角色权限：管理员=全体；教师=本班学生；学生=对应学科教师。"""
    if me.role == "admin":
        return True
    if me.role == "teacher":
        if other.role != "student":
            return False
        if not me.class_name:  # 未分班教师可联系所有学生
            return True
        return other.class_name == me.class_name
    if me.role == "student":
        if other.role != "teacher":
            return False
        my_sids = await _subject_ids(db, me.id)
        if not my_sids:  # 未分配学科学生可联系所有教师
            return True
        other_sids = await _subject_ids(db, other.id)
        return bool(my_sids & other_sids)
    return False


async def _contacts_payload(users: list[User], db: AsyncSession) -> list[dict]:
    """批量装载学科名，返回联系人列表。"""
    ids = [u.id for u in users]
    subject_names: dict[int, list[str]] = {u: [] for u in ids}
    if ids:
        res = await db.execute(
            select(UserSubject.user_id, Subject.name)
            .join(Subject, Subject.id == UserSubject.subject_id)
            .where(UserSubject.user_id.in_(ids))
        )
        for uid, name in res.all():
            subject_names.setdefault(uid, []).append(name)

    return [
        {
            "id": u.id,
            "username": u.username,
            "real_name": u.real_name,
            "role": u.role,
            "class_name": u.class_name,
            "subjects": subject_names.get(u.id, []),
        }
        for u in users
    ]


@router.get("/contacts")
async def list_contacts(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """可私聊的联系人列表（按角色）：管理员=全体；教师=本班学生；学生=对应学科教师。"""
    me = await _get_user(db, current_user["user_id"])
    if not me:
        raise HTTPException(404, "用户不存在")

    if me.role == "admin":
        stmt = select(User).where(User.id != me.id).order_by(User.role, User.class_name, User.id)
        users = (await db.execute(stmt)).scalars().all()
    elif me.role == "teacher":
        stmt = select(User).where(User.role == "student", User.id != me.id)
        if me.class_name:
            stmt = stmt.where(User.class_name == me.class_name)
        users = (await db.execute(stmt.order_by(User.class_name, User.id))).scalars().all()
    else:  # student
        teachers = (await db.execute(
            select(User).where(User.role == "teacher", User.id != me.id).order_by(User.id)
        )).scalars().all()
        my_sids = await _subject_ids(db, me.id)
        if my_sids:
            filtered = []
            for t in teachers:
                if my_sids & await _subject_ids(db, t.id):
                    filtered.append(t)
            teachers = filtered
        users = teachers

    return await _contacts_payload(users, db)


@router.get("/conversations")
async def list_conversations(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """列出与我有过消息的对方 + 最后一条消息 + 未读数。"""
    me = current_user["user_id"]
    result = await db.execute(
        select(Message)
        .where(or_(Message.sender_id == me, Message.recipient_id == me))
        .order_by(Message.created_at.desc())
        .limit(2000)
    )
    msgs = result.scalars().all()

    agg: dict[int, dict] = {}
    for m in msgs:
        other = m.recipient_id if m.sender_id == me else m.sender_id
        entry = agg.get(other)
        if entry is None:
            entry = {"last": m, "unread": 0}
            agg[other] = entry
        if m.recipient_id == me and not m.is_read:
            entry["unread"] += 1

    # 对方用户信息
    other_ids = list(agg.keys())
    users = {}
    if other_ids:
        ures = await db.execute(select(User).where(User.id.in_(other_ids)))
        for u in ures.scalars().all():
            users[u.id] = u

    items = []
    for other, entry in agg.items():
        m = entry["last"]
        items.append({
            "user": _user_brief(users.get(other)),
            "last_message": m.content,
            "last_time": m.created_at.isoformat() if m.created_at else None,
            "unread": entry["unread"],
        })
    # 按最后消息时间倒序（agg 已按倒序遍历，保持顺序即可）
    return items


@router.get("/with/{user_id}")
async def list_with_user(
    user_id: int,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """与某用户的私信记录，并把对方发来的置为已读。"""
    me = current_user["user_id"]
    cond = or_(
        (Message.sender_id == me) & (Message.recipient_id == user_id),
        (Message.sender_id == user_id) & (Message.recipient_id == me),
    )
    base = select(Message).where(cond)
    total = await db.scalar(select(func.count()).select_from(base.subquery()))

    result = await db.execute(
        base.order_by(Message.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
    )
    msgs = result.scalars().all()

    # 标记对方发来的为已读
    updated = False
    for m in msgs:
        if m.recipient_id == me and not m.is_read:
            m.is_read = True
            updated = True
    if updated:
        await db.commit()

    other = (await db.execute(select(User).where(User.id == user_id))).scalar_one_or_none()
    return {
        "other": _user_brief(other),
        "data": [
            {
                "id": m.id,
                "sender_id": m.sender_id,
                "recipient_id": m.recipient_id,
                "content": m.content,
                "is_read": m.is_read,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in reversed(msgs)  # 按时间正序返回
        ],
        "total": total or 0,
        "page": page,
        "page_size": page_size,
    }


@router.post("")
async def send_message(
    body: SendRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """发送私信（按角色校验收件人权限）。"""
    me = current_user["user_id"]
    if body.recipient_id == me:
        raise HTTPException(400, "不能给自己发消息")
    recipient = await _get_user(db, body.recipient_id)
    if not recipient:
        raise HTTPException(404, "收件人不存在")

    me_user = await _get_user(db, me)
    if me_user and not await _can_message(db, me_user, recipient):
        raise HTTPException(403, "根据你的角色，不能给该用户发送私信")

    msg = Message(sender_id=me, recipient_id=body.recipient_id, content=body.content)
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return {
        "id": msg.id,
        "sender_id": msg.sender_id,
        "recipient_id": msg.recipient_id,
        "content": msg.content,
        "is_read": msg.is_read,
        "created_at": msg.created_at.isoformat() if msg.created_at else None,
    }


@router.get("/unread-count")
async def unread_count(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """未读私信总数（用于红点）。"""
    me = current_user["user_id"]
    n = await db.scalar(
        select(func.count(Message.id)).where(Message.recipient_id == me, Message.is_read.is_(False))
    )
    return {"count": n or 0}
