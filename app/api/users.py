"""User management API endpoints — CRUD, batch import, profile."""
from fastapi import APIRouter, Depends, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_admin_user, get_current_user, hash_password
from app.models import User as UserModel, UserSubject, Subject
from app.schemas.schemas import (
    ImportResult,
    PaginatedResponse,
    UserCreate,
    UserOut,
    UserUpdate,
)


class SubjectIdsBody(BaseModel):
    subject_ids: list[int]


router = APIRouter(prefix="/api/users", tags=["users"])


@router.get("", response_model=PaginatedResponse)
async def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=10000),
    role: str | None = None,
    class_name: str | None = None,
    search: str | None = None,
    subject_id: int | None = None,
    online_status: str | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List users with optional filters. Teachers see their class only."""
    query = select(UserModel)

    # Role-based visibility
    if current_user["role"] == "teacher":
        query = query.where(UserModel.role == "student")
    elif current_user["role"] == "student":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权查看用户列表")

    if role:
        query = query.where(UserModel.role == role)
    if class_name:
        query = query.where(UserModel.class_name == class_name)
    if search:
        query = query.where(
            (UserModel.username.ilike(f"%{search}%"))
            | (UserModel.real_name.ilike(f"%{search}%"))
            | (UserModel.student_id.ilike(f"%{search}%"))
        )
    if subject_id:
        subq = select(UserSubject.user_id).where(UserSubject.subject_id == subject_id)
        query = query.where(UserModel.id.in_(subq))

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Fetch page
    offset = (page - 1) * page_size
    query = query.order_by(UserModel.id.asc()).offset(offset).limit(page_size)
    result = await db.execute(query)
    users = result.scalars().all()

    # Batch-load all subjects for these users (avoid N+1)
    user_ids = [u.id for u in users]
    subject_map: dict[int, list[dict]] = {uid: [] for uid in user_ids}
    if user_ids:
        from sqlalchemy import select as sa_select
        subj_result = await db.execute(
            sa_select(UserSubject.user_id, Subject.id, Subject.name)
            .join(Subject, Subject.id == UserSubject.subject_id)
            .where(UserSubject.user_id.in_(user_ids))
        )
        for user_id, subj_id, subj_name in subj_result.all():
            subject_map.setdefault(user_id, []).append({"id": subj_id, "name": subj_name})

    # Enrich with subjects
    user_list = []
    for u in users:
        u_dict = UserOut.model_validate(u).model_dump()
        u_dict["subjects"] = subject_map.get(u.id, [])
        user_list.append(u_dict)

    return PaginatedResponse(
        items=user_list,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=max(1, (total + page_size - 1) // page_size),
    )


class UserCreateBody(UserCreate):
    subject_ids: list[int] = []


@router.post("", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreateBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """手动创建单个用户（管理员）。"""
    # 所有用户统一用学工号登录（用户名 = 学工号）
    username = data.student_id or data.username
    if not username:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="请填写学工号")

    existing = await db.execute(select(UserModel).where(UserModel.username == username))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="学工号已存在")

    user = UserModel(
        username=username,
        password_hash=hash_password(data.password),
        real_name=data.real_name,
        role=data.role,
        class_name=data.class_name,
        student_id=data.student_id,
        email=data.email,
    )
    db.add(user)
    await db.flush()  # 拿到 user.id

    for sid in data.subject_ids:
        db.add(UserSubject(user_id=user.id, subject_id=sid))

    await db.commit()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.get("/classes")
async def list_classes(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """List distinct class names for filtering."""
    result = await db.execute(
        select(UserModel.class_name).where(UserModel.class_name.isnot(None)).distinct()
    )
    classes = [r[0] for r in result.all() if r[0]]
    classes.sort()
    return classes


@router.get("/{user_id}", response_model=UserOut)
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get a single user by ID."""
    if current_user["role"] == "student" and current_user["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权查看其他用户信息")

    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    return UserOut.model_validate(user)


@router.put("/{user_id}", response_model=UserOut)
async def update_user(
    user_id: int,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Update user info. Admin-only for role/status changes."""
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

    # Permission check
    if current_user["role"] != "admin" and current_user["user_id"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="无权修改此用户")
    if current_user["role"] != "admin":
        # 非管理员只能改自己的基础信息（姓名等），不能改角色/状态/班级/学工号
        if data.role is not None or data.is_active is not None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只有管理员可以修改角色或状态")
        if data.class_name is not None or data.student_id is not None:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="只有管理员可以修改班级或学工号")

    update_data = data.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    await db.flush()
    await db.refresh(user)
    return UserOut.model_validate(user)


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Delete a user (admin only)."""
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")
    if user.id == current_user["user_id"]:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="不能删除自己")

    await db.delete(user)
    return {"message": f"已删除用户: {user.username}"}


@router.get("/{user_id}/subjects")
async def get_user_subjects(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user),
):
    """Get subjects assigned to a user."""
    result = await db.execute(
        select(Subject).join(UserSubject).where(UserSubject.user_id == user_id)
    )
    return [{"id": s.id, "name": s.name} for s in result.scalars().all()]


@router.put("/{user_id}/subjects")
async def update_user_subjects(
    user_id: int,
    data: "SubjectIdsBody",
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Update subjects assigned to a user (admin only). Body: {"subject_ids": [1,2,3]}"""
    # Remove existing
    await db.execute(delete(UserSubject).where(UserSubject.user_id == user_id))
    # Add new
    for sid in data.subject_ids:
        db.add(UserSubject(user_id=user_id, subject_id=sid))
    await db.flush()
    return {"message": "学科分配已更新", "subject_ids": data.subject_ids}


@router.post("/{user_id}/reset-password")
async def reset_user_password(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Reset a user's password to their student_id or default."""
    result = await db.execute(select(UserModel).where(UserModel.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="用户不存在")

    new_password = user.student_id or "123456"
    user.password_hash = hash_password(new_password)
    await db.flush()
    return {"message": f"密码已重置为: {new_password}"}


@router.post("/import-excel", response_model=ImportResult)
async def import_users_excel(
    file: UploadFile,
    subject_id: int = Form(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """批量导入学生（仅学生，自动分配到指定学科）。

    按列名自动识别：姓名 / 学号 / 班级。
    支持 .xls / .xlsx / .xlsm / .csv。
    """
    import io
    import pandas as pd

    filename = (file.filename or "").lower()
    contents = await file.read()

    def _read_df():
        if filename.endswith(".xls"):
            return pd.read_excel(io.BytesIO(contents), engine="xlrd", dtype=str)
        if filename.endswith((".xlsx", ".xlsm")):
            return pd.read_excel(io.BytesIO(contents), engine="openpyxl", dtype=str)
        if filename.endswith(".csv"):
            for enc in ("utf-8-sig", "gbk", "utf-8"):
                try:
                    return pd.read_csv(io.BytesIO(contents), encoding=enc, dtype=str)
                except UnicodeDecodeError:
                    continue
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="无法识别 CSV 编码")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="仅支持 .xls / .xlsx / .xlsm / .csv 文件")

    try:
        df = _read_df()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"无法解析表格: {e}")

    def _find_col(*keywords):
        for col in df.columns:
            c = str(col)
            if any(k in c for k in keywords):
                return col
        return None

    col_name = _find_col("姓名", "名字")
    col_sid = _find_col("学号", "学工号")
    col_class = _find_col("班级")

    if col_name is None or col_sid is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="表格缺少「姓名」或「学工号」列")

    def _clean(v):
        s = str(v if v is not None else "").strip()
        return "" if s.lower() in ("nan", "none", "nat") else s

    success = 0
    total = 0
    errors = []

    for i, row in df.iterrows():
        name = _clean(row[col_name])
        sid = _clean(row[col_sid])
        if not name or not sid:
            continue
        if sid.endswith(".0"):
            sid = sid[:-2]
        total += 1

        class_name = _clean(row[col_class]) if col_class else ""

        username = sid  # 学工号作为登录名
        existing = await db.scalar(select(UserModel).where(UserModel.username == username))
        if existing:
            errors.append(f"第{i + 2}行: 学工号'{sid}'已存在")
            continue
        try:
            user = UserModel(
                username=username,
                password_hash=hash_password(sid),   # 默认密码 = 学工号
                real_name=name,
                role="student",                      # 只能是学生
                class_name=class_name or None,
                student_id=sid,
            )
            db.add(user)
            await db.flush()
            db.add(UserSubject(user_id=user.id, subject_id=subject_id))
            await db.commit()   # 每个学生提交一次，避免长时间占用 SQLite 写锁
            success += 1
        except Exception as e:
            await db.rollback()  # 清除失败事务（如数据库锁），继续下一个
            errors.append(f"第{i + 2}行: {str(e)}")

    return ImportResult(total=total, success=success, failed=total - success, errors=errors)
