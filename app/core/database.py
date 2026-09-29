"""Async SQLAlchemy engine and session management."""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

_is_sqlite = settings.DATABASE_URL.startswith("sqlite")

_engine_kwargs = {"echo": settings.DEBUG}
if not _is_sqlite:
    _engine_kwargs.update(
        pool_size=settings.DATABASE_POOL_SIZE,
        max_overflow=settings.DATABASE_MAX_OVERFLOW,
    )

engine = create_async_engine(settings.DATABASE_URL, **_engine_kwargs)

if _is_sqlite:
    from sqlalchemy import event

    @event.listens_for(engine.sync_engine, "connect")
    def _sqlite_pragma(dbapi_conn, connection_record):
        # 写锁忙等 30s，避免并发写时报 "database is locked"
        cur = dbapi_conn.cursor()
        try:
            cur.execute("PRAGMA busy_timeout = 30000")
        finally:
            cur.close()

async_session_factory = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models."""
    pass


async def get_db() -> AsyncSession:  # type: ignore
    """FastAPI dependency: yields an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Create all tables (for development / first run)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if _is_sqlite:
            await conn.run_sync(_add_missing_columns)
            await conn.run_sync(_migrate_usernames)


def _add_missing_columns(sync_conn):
    """为已有 SQLite 库补齐新增列（create_all 不会给已存在的表加列）。"""
    from sqlalchemy import text
    for table, col, col_type in (
        ("knowledge_points", "subject_id", "INTEGER"),
        ("document_subjects", "in_kb", "INTEGER"),
        ("document_subjects", "in_qb", "INTEGER"),
        ("document_subjects", "qb_chapter", "VARCHAR(255)"),
        ("subjects", "example_questions", "TEXT"),
        ("messages", "deleted_by_sender", "INTEGER"),
        ("messages", "deleted_by_recipient", "INTEGER"),
        ("test_questions", "source", "VARCHAR(20)"),
        ("test_questions", "source_doc_id", "INTEGER"),
        ("test_questions", "page_number", "INTEGER"),
        ("test_questions", "original_answer", "TEXT"),
        ("test_questions", "llm_corrected", "INTEGER"),
        ("test_questions", "llm_verified", "INTEGER"),
        ("test_questions", "extract_version", "INTEGER"),
    ):
        existing = {row[1] for row in sync_conn.execute(text(f"PRAGMA table_info({table})"))}
        if col not in existing:
            sync_conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))


def _migrate_usernames(sync_conn):
    """迁移：所有用户的登录名统一为学工号（username = student_id）。

    早期手动添加的用户可能 username 与学工号不一致，登录时会用学工号查 username，
    导致登不上。这里把有学工号的用户 username 一律改成学工号。
    """
    from sqlalchemy import text
    sync_conn.execute(text(
        "UPDATE users SET username = student_id "
        "WHERE student_id IS NOT NULL AND student_id != '' AND username != student_id"
    ))
