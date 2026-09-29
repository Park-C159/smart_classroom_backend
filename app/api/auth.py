"""Authentication API endpoints: login, register, refresh, logout."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.security import (
    bearer_scheme,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.core.redis_client import blacklist_token, is_token_blacklisted
from app.models import User
from app.schemas.schemas import LoginRequest, RefreshRequest, TokenResponse, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return JWT token pair."""
    result = await db.execute(select(User).where(User.username == request.username))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户名或密码错误")

    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="账号已被禁用")

    token_data = {"sub": str(user.id), "username": user.username, "role": user.role}
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        user=UserOut.model_validate(user),
    )


class RegisterBody(BaseModel):
    student_id: str = Field(..., min_length=1, max_length=50, description="学工号（登录账号）")
    password: str = Field(..., min_length=6, max_length=128)
    real_name: str = Field(..., min_length=1, max_length=50)
    class_name: str | None = None


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterBody,
    db: AsyncSession = Depends(get_db),
):
    """学生自助注册：用学工号 + 密码注册账号（仅学生，公开接口）。"""
    username = request.student_id.strip()

    existing = await db.execute(select(User).where(User.username == username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="该学工号已注册")

    user = User(
        username=username,
        password_hash=hash_password(request.password),
        real_name=request.real_name,
        role="student",
        class_name=request.class_name,
        student_id=username,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(request: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """Refresh access token using a valid refresh token."""
    try:
        payload = decode_token(request.refresh_token)
        if payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌类型无效")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="刷新令牌无效或已过期")

    if await is_token_blacklisted(request.refresh_token):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="令牌已被撤销")

    user_id = int(payload["sub"])
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在或已禁用")

    await blacklist_token(request.refresh_token, settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400)

    token_data = {"sub": str(user.id), "username": user.username, "role": user.role}
    return TokenResponse(
        access_token=create_access_token(token_data),
        refresh_token=create_refresh_token(token_data),
        user=UserOut.model_validate(user),
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    current_user: dict = Depends(get_current_user),
):
    """Logout: blacklist the current access token."""
    await blacklist_token(credentials.credentials, settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return {"message": "已退出登录"}


@router.get("/me", response_model=UserOut)
async def get_me(current_user: dict = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Get current authenticated user's profile."""
    result = await db.execute(select(User).where(User.id == current_user["user_id"]))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return UserOut.model_validate(user)
