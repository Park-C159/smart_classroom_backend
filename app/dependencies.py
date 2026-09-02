"""FastAPI dependencies: database session, current user, role checks."""
from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user, get_teacher_or_admin
from app.models import User

__all__ = [
    "get_db",
    "get_current_user",
    "get_admin_user",
    "get_teacher_or_admin",
    "get_user_optional",
]


async def get_user_optional(
    current_user: dict | None = Depends(get_current_user),
) -> dict | None:
    """Like get_current_user but returns None instead of 401 when unauthenticated."""
    return current_user
