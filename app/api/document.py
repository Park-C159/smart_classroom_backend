"""Document management API — PDF upload, MinerU processing, page retrieval."""
import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from fastapi.responses import FileResponse, RedirectResponse
from sqlalchemy import func, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.core.database import get_db
from app.core.security import get_current_user, get_teacher_or_admin
from app.models import Document, Subject, DocumentSubject, User, QuestionBank, KnowledgePoint
from app.services.document_processor import DocumentProcessor
from app.services.rag_service import RAGService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/documents", tags=["文档管理"])
processor = DocumentProcessor()

DATA_DIR = Path(settings.DATA_DIR)


# Track cancelled document IDs so background tasks can abort
_cancelled_docs: set[int] = set()


def _run_mineru_blocking(pdf_path: Path, doc_id: int):
    """Run MinerU in a thread to avoid blocking the event loop.

    MinerU 跑在独立进程（常驻 mineru-api 或临时子进程），不占用后端 GPU，
    因此无需 clear_gpu/restore_defaults（避免与并发 RAG 检索抢模型状态）。
    """
    return processor.process_pdf(pdf_path, doc_id)


def _append_log(doc_id: int, msg: str):
    """Append a timestamped message to the mineru log for this document."""
    import datetime
    log_file = DATA_DIR / "parsed" / str(doc_id) / "mineru.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    try:
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {msg}\n")
    except Exception:
        pass


async def process_document_task(doc_id: int, pdf_path: Path):
    """Background task: MinerU parse → knowledge tree → FAISS index build.

    GPU-heavy MinerU runs in a thread pool so the event loop stays responsive
    for other requests during parsing.  Progress written to DB and log file
    so the frontend can show live status.
    """
    import asyncio
    from app.core.database import async_session_factory

    _cancelled_docs.discard(doc_id)  # fresh start, clear any stale cancel flag
    _append_log(doc_id, "📋 后台任务已启动")

    def _is_cancelled():
        return doc_id in _cancelled_docs

    async with async_session_factory() as db:
        if _is_cancelled(): return
        try:
            doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            if doc:
                doc.status = "processing"
                doc.progress = 0
                await db.commit()
            _append_log(doc_id, "状态: processing (0%)")
        except Exception as e:
            logger.error("Document processing failed: %s", e)
            return

    # Step 1: MinerU PDF parsing (GPU-heavy, runs in thread pool)
    if _is_cancelled():
        _append_log(doc_id, "🛑 任务被取消")
        return
    _append_log(doc_id, "⏳ 正在将答疑模型移出 GPU...")
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(None, _run_mineru_blocking, pdf_path, doc_id)
        _append_log(doc_id, f"✅ MinerU 解析完成，共 {result.get('total_pages', 0)} 页")
    except Exception as e:
        logger.error("MinerU parsing failed: %s", e)
        _append_log(doc_id, f"❌ MinerU 解析失败: {e}")
        if _is_cancelled():
            _append_log(doc_id, "🛑 任务已被取消，跳过后续步骤")
            return
        async with async_session_factory() as db:
            doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            if doc:
                doc.status = "failed"
                doc.progress = 0
                await db.commit()
        return

    if _is_cancelled():
        _append_log(doc_id, "🛑 任务被取消，跳过后续步骤")
        return

    # Steps 2-3: DB work + knowledge tree + FAISS (async-safe, RAG models on GPU)
    async with async_session_factory() as db:
        try:
            if _is_cancelled():
                _append_log(doc_id, "🛑 任务被取消")
                return
            doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            if doc:
                doc.status = "parsed"
                doc.total_pages = result.get("total_pages", 0)
                doc.progress = 33
                await db.commit()
            _append_log(doc_id, "状态: parsed (33%) — 构建知识树...")
            try:
                from app.services.knowledge_tree_service import KnowledgeTreeService
                from app.services.llm_service import LLMService
                kts = KnowledgeTreeService()
                _append_log(doc_id, "🔍 规则引擎结构化...")
                kt_result = await kts.build_from_parsed(doc_id, result, db)
                await db.commit()

                _append_log(doc_id, "🤖 调用 LLM 生成知识点摘要...")
                llm = LLMService()
                await kts.generate_kp_summaries(db, llm)
                await db.commit()
                logger.info("Knowledge tree built for doc %d: %s", doc_id, kt_result)
                _append_log(doc_id, f"✅ 知识树构建完成: {kt_result}")
            except Exception as e:
                logger.warning("Knowledge tree build failed (non-fatal): %s", e)
                _append_log(doc_id, f"⚠️ 知识树构建失败 (非致命): {e}")

            doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            if doc:
                doc.progress = 66
                await db.commit()
            _append_log(doc_id, "状态: 66% — 构建 FAISS 索引...")

            # Step 3: Build FAISS indices
            try:
                _append_log(doc_id, "📊 正在移动模型到 GPU 并构建向量索引...")
                rag = RAGService()
                rag.build_index(doc_id)
                await rag.build_knowledge_indices(doc_id)
                logger.info("RAG indices built for doc %d", doc_id)
                _append_log(doc_id, "✅ FAISS 索引构建完成")
            except Exception as e:
                logger.warning("RAG index build failed: %s", e)
                _append_log(doc_id, f"⚠️ 索引构建失败: {e}")

            doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
            if doc:
                doc.status = "completed"
                doc.progress = 100
                await db.commit()
            _append_log(doc_id, "🎉 全部完成! 状态: completed (100%)")

        except Exception as e:
            logger.error("Document processing failed: %s", e)
            _append_log(doc_id, f"❌ 处理失败: {e}")
            try:
                doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
                if doc:
                    doc.status = "failed"
                    doc.progress = 0
                    await db.commit()
            except Exception:
                pass


@router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    subject_ids: Optional[str] = None,  # comma-separated subject IDs, e.g. "1,2"
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Upload PDF for MinerU processing. Optionally assign to subjects via subject_ids."""
    if not file.filename or not file.filename.endswith(".pdf"):
        raise HTTPException(400, "仅支持PDF文件")

    file_id = str(uuid.uuid4())
    pdf_dir = DATA_DIR / "pdfs"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    save_path = pdf_dir / f"{file_id}.pdf"

    content = await file.read()
    with open(save_path, "wb") as f:
        f.write(content)

    doc = Document(
        title=title or file.filename,
        filename=file.filename,
        file_path=str(save_path),
        upload_by=int(current_user["user_id"]),
        status="pending",
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    # Optionally assign to subjects
    if subject_ids:
        try:
            sids = [int(x.strip()) for x in subject_ids.split(",") if x.strip()]
            for sid in sids:
                db.add(DocumentSubject(document_id=doc.id, subject_id=sid))
            await db.commit()
        except ValueError:
            pass  # ignore invalid IDs

    background_tasks.add_task(process_document_task, doc.id, save_path)

    return {
        "id": doc.id, "title": doc.title, "filename": doc.filename,
        "status": doc.status, "total_pages": doc.total_pages,
        "progress": doc.progress, "created_at": doc.created_at.isoformat() if doc.created_at else None,
    }


@router.get("/")
async def list_documents(
    subject_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """List all documents, optionally filtered by subject."""
    query = select(Document).order_by(Document.created_at.desc())
    if subject_id:
        query = query.join(DocumentSubject).where(DocumentSubject.subject_id == subject_id)
    result = await db.execute(query)
    docs = result.scalars().all()

    # Batch-load all subjects for these documents (avoid N+1)
    doc_ids = [d.id for d in docs]
    subject_map: dict[int, list[dict]] = {did: [] for did in doc_ids}
    if doc_ids:
        from sqlalchemy import select as sa_select
        subj_result = await db.execute(
            sa_select(DocumentSubject.document_id, Subject.id, Subject.name)
            .join(Subject, Subject.id == DocumentSubject.subject_id)
            .where(DocumentSubject.document_id.in_(doc_ids))
        )
        for doc_id, subj_id, subj_name in subj_result.all():
            subject_map.setdefault(doc_id, []).append({"id": subj_id, "name": subj_name})

    doc_list = []
    for d in docs:
        doc_list.append({
            "id": d.id, "title": d.title, "filename": d.filename,
            "doc_type": d.doc_type, "is_primary": d.is_primary,
            "subject_id": d.subject_id,
            "status": d.status, "total_pages": d.total_pages,
            "progress": d.progress, "upload_by": d.upload_by,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "subjects": subject_map.get(d.id, []),
        })

    return doc_list


# ── PDF Viewer ──

@router.get("/{doc_id}/pdf")
async def view_document_pdf(
    doc_id: int,
    page: int = Query(default=0, description="Page number to jump to"),
    db: AsyncSession = Depends(get_db),
):
    """Serve PDF file with optional page fragment for browser viewer."""
    doc = await db.scalar(select(Document).where(Document.id == doc_id))
    if not doc or not doc.file_path:
        raise HTTPException(404, "文档不存在或PDF路径无效")
    pdf_path = Path(doc.file_path)
    if not pdf_path.exists():
        raise HTTPException(404, "PDF文件不存在")
    # Use fragment to jump to page: #page=N
    if page > 0:
        return RedirectResponse(f"/api/documents/{doc_id}/pdf/view#page={page}")
    return FileResponse(str(pdf_path), media_type="application/pdf")


@router.get("/{doc_id}/pdf/view")
async def serve_pdf(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Serve raw PDF file for browser viewing."""
    doc = await db.scalar(select(Document).where(Document.id == doc_id))
    if not doc or not doc.file_path:
        raise HTTPException(404, "文档不存在")
    pdf_path = Path(doc.file_path)
    if not pdf_path.exists():
        raise HTTPException(404, "PDF文件不存在")
    return FileResponse(str(pdf_path), media_type="application/pdf",
                        headers={"Content-Disposition": "inline; filename=\"textbook.pdf\""})


# ── QuestionBanks ──

@router.get("/exercises")
async def list_exercises(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    kp_id: str | None = None,
    source: str | None = None,
    question_type: str | None = None,
    difficulty: int | None = None,
    has_answer: bool | None = None,
    db: AsyncSession = Depends(get_db),
):
    """List exercises with optional filters."""
    conditions = []
    if kp_id:
        conditions.append(QuestionBank.kp_id == kp_id)
    if source:
        conditions.append(QuestionBank.source == source)
    if question_type:
        conditions.append(QuestionBank.question_type == question_type)
    if difficulty:
        conditions.append(QuestionBank.difficulty == difficulty)
    if has_answer is True:
        conditions.append(QuestionBank.answer_text.isnot(None))
        conditions.append(QuestionBank.answer_text != "")
    elif has_answer is False:
        conditions.append(or_(QuestionBank.answer_text.is_(None), QuestionBank.answer_text == ""))

    base = select(QuestionBank)
    if conditions:
        base = base.where(and_(*conditions))

    total = await db.scalar(select(func.count()).select_from(base.subquery()))
    offset = (page - 1) * page_size
    result = await db.execute(
        base.order_by(QuestionBank.id.desc()).offset(offset).limit(page_size)
    )
    exercises = result.scalars().all()

    # Load document titles for exercises with source_doc_id
    doc_ids = list(set(e.source_doc_id for e in exercises if e.source_doc_id))
    doc_titles = {}
    if doc_ids:
        doc_result = await db.execute(select(Document.id, Document.title).where(Document.id.in_(doc_ids)))
        doc_titles = {d[0]: d[1] for d in doc_result.all()}

    return {
        "data": [
            {
                "id": e.id, "kp_id": e.kp_id, "question_text": e.question_text,
                "answer_text": e.answer_text, "question_type": e.question_type,
                "difficulty": e.difficulty, "source": e.source,
                "page_number": e.page_number,
                "source_doc_id": e.source_doc_id,
                "source_doc_title": doc_titles.get(e.source_doc_id, ""),
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in exercises
        ],
        "total": total or 0, "page": page, "page_size": page_size,
    }


@router.get("/exercises/stats")
async def get_exercise_stats(db: AsyncSession = Depends(get_db)):
    """Get exercise statistics."""
    total = await db.scalar(select(func.count(QuestionBank.id)))
    textbook = await db.scalar(select(func.count(QuestionBank.id)).where(QuestionBank.source == "textbook"))
    teacher = await db.scalar(select(func.count(QuestionBank.id)).where(QuestionBank.source == "teacher"))
    choice = await db.scalar(select(func.count(QuestionBank.id)).where(QuestionBank.question_type == "choice"))
    calc = await db.scalar(select(func.count(QuestionBank.id)).where(QuestionBank.question_type.in_(["calculation", "proof"])))
    with_answer = await db.scalar(
        select(func.count(QuestionBank.id)).where(
            and_(QuestionBank.answer_text.isnot(None), QuestionBank.answer_text != "")
        )
    )

    return {
        "total": total or 0,
        "textbook": textbook or 0,
        "teacher": teacher or 0,
        "choice": choice or 0,
        "calculation_proof": calc or 0,
        "with_answer": with_answer or 0,
    }


@router.get("/exercises/{exercise_id}")
async def get_exercise_detail(
    exercise_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get single exercise with full details."""
    result = await db.execute(select(QuestionBank).where(QuestionBank.id == exercise_id))
    e = result.scalar_one_or_none()
    if not e:
        raise HTTPException(404, "习题不存在")

    doc_title = ""
    if e.source_doc_id:
        doc_result = await db.execute(select(Document.title).where(Document.id == e.source_doc_id))
        doc_title = doc_result.scalar_one_or_none() or ""

    return {
        "id": e.id, "kp_id": e.kp_id, "question_text": e.question_text,
        "answer_text": e.answer_text, "question_type": e.question_type,
        "difficulty": e.difficulty, "source": e.source,
        "page_number": e.page_number,
        "source_doc_id": e.source_doc_id,
        "source_doc_title": doc_title,
        "images": e.images,
        "embedding_text": e.embedding_text,
        "verified": e.verified,
        "created_at": e.created_at.isoformat() if e.created_at else None,
    }


# ── QuestionBank CRUD ──

class QuestionBankCreateBody(BaseModel):
    kp_id: str | None = None
    chapter: str | None = None
    question_text: str
    answer_text: str | None = None
    question_type: str = "calculation"
    difficulty: int = 3
    page_number: int | None = None
    subject_id: int | None = None
    images: list[dict] | None = None


class QuestionBankUpdateBody(BaseModel):
    kp_id: str | None = None
    question_text: str | None = None
    answer_text: str | None = None
    question_type: str | None = None
    difficulty: int | None = None
    page_number: int | None = None
    images: list[dict] | None = None
    verified: bool | None = None


@router.post("/exercises")
async def create_exercise(
    data: QuestionBankCreateBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Manually create an exercise (teacher/admin)."""
    from app.services.vlm_service import VLMService
    # Process images with VLM if any
    images = data.images or []
    embedding_parts = [data.question_text]
    if images:
        vlm = VLMService()
        for img in images:
            path = img.get("path", "")
            if path and Path(path).exists():
                result = vlm.process_image(path)
                img["vlm_desc"] = result.get("description")
                if result.get("usable"):
                    embedding_parts.append(result["description"])

    # 章节：优先用前端传的 chapter，否则从 kp_id 反推
    chapter = data.chapter
    if not chapter and data.kp_id:
        kp = await db.get(KnowledgePoint, data.kp_id)
        if kp and kp.chapter:
            chapter = kp.chapter

    ex = QuestionBank(
        kp_id=data.kp_id,
        chapter=chapter,
        question_text=data.question_text,
        answer_text=data.answer_text,
        question_type=data.question_type,
        difficulty=data.difficulty,
        page_number=data.page_number,
        subject_id=data.subject_id or 9,
        source="teacher",
        images=images if images else None,
        embedding_text="\n".join(embedding_parts),
    )
    db.add(ex)
    await db.commit()
    await db.refresh(ex)
    return {"id": ex.id, "message": "习题已创建"}


@router.put("/exercises/{exercise_id}")
async def update_exercise(
    exercise_id: int,
    data: QuestionBankUpdateBody,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Update an exercise (teacher/admin)."""
    result = await db.execute(select(QuestionBank).where(QuestionBank.id == exercise_id))
    ex = result.scalar_one_or_none()
    if not ex:
        raise HTTPException(404, "习题不存在")

    if data.kp_id is not None: ex.kp_id = data.kp_id
    if data.question_text is not None: ex.question_text = data.question_text
    if data.answer_text is not None: ex.answer_text = data.answer_text
    if data.question_type is not None: ex.question_type = data.question_type
    if data.difficulty is not None: ex.difficulty = data.difficulty
    if data.page_number is not None: ex.page_number = data.page_number
    if data.images is not None: ex.images = data.images
    if data.verified is not None: ex.verified = data.verified
    # Rebuild embedding_text if question or images changed
    if data.question_text is not None or data.images is not None:
        parts = [ex.question_text]
        if ex.images:
            for img in ex.images:
                if img.get("vlm_desc"):
                    parts.append(img["vlm_desc"])
        ex.embedding_text = "\n".join(parts)

    await db.commit()
    return {"id": ex.id, "message": "习题已更新"}


@router.delete("/exercises/{exercise_id}")
async def delete_exercise(
    exercise_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Hard delete an exercise (teacher/admin)."""
    result = await db.execute(select(QuestionBank).where(QuestionBank.id == exercise_id))
    ex = result.scalar_one_or_none()
    if not ex:
        raise HTTPException(404, "习题不存在")

    await db.delete(ex)
    await db.commit()
    return {"message": "习题已删除"}


@router.post("/exercises/upload-image")
async def upload_exercise_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Upload an image for an exercise. Returns the path and VLM description."""
    upload_dir = Path(settings.DATA_DIR) / "exercise_images"
    upload_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix if file.filename else ".png"
    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = upload_dir / filename

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    # Run VLM
    from app.services.vlm_service import VLMService
    vlm = VLMService()
    vr = vlm.process_image(str(filepath))

    return {
        "path": str(filepath),
        "type": vr["type"],
        "vlm_desc": vr.get("description"),
        "usable": vr.get("usable", False),
    }


@router.get("/{doc_id}")
async def get_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")
    return {
        "id": doc.id, "title": doc.title, "filename": doc.filename,
        "status": doc.status, "total_pages": doc.total_pages,
        "progress": doc.progress, "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "file_path": doc.file_path,
    }


@router.get("/{doc_id}/parsed")
async def get_parsed_result(doc_id: int, db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)):
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")
    if doc.status in ("pending", "processing"):
        return {"status": doc.status, "message": "文档正在处理中", "progress": doc.progress}
    if doc.status == "failed":
        return {"status": doc.status, "message": "文档处理失败"}
    result = processor.get_parsed_result(doc_id)
    return {"status": doc.status, "data": result}


@router.get("/{doc_id}/page/{page_num}")
async def get_page_content(doc_id: int, page_num: int, db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)):
    try:
        rag = RAGService()
        result = rag.get_page_content(doc_id, page_num)
        return result
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"获取页面内容失败: {e}")


@router.get("/{doc_id}/log")
async def get_mineru_log(doc_id: int, db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin)):
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")
    log_file = DATA_DIR / "parsed" / str(doc_id) / "mineru.log"
    log_content = ""
    if log_file.exists():
        try:
            log_content = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            log_content = f"读取日志失败: {e}"
    else:
        log_content = "日志文件尚未生成"
    return {"doc_id": doc_id, "status": doc.status, "log": log_content[:200000],
            "updated_at": datetime.now(timezone.utc).isoformat()}


@router.delete("/{doc_id}")
async def delete_document(doc_id: int, db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin)):
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")

    # Signal any running background task to stop
    _cancelled_docs.add(doc_id)
    logger.info("🛑 取消文档 %d 的后台解析任务", doc_id)

    # Delete original PDF
    try:
        pdf_path = Path(doc.file_path)
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception as e:
        logger.warning("删除 PDF 文件失败: %s", e)

    # Delete parsed output directory
    parsed_dir = DATA_DIR / "parsed" / str(doc_id)
    if parsed_dir.exists():
        try:
            shutil.rmtree(parsed_dir)
        except Exception as e:
            logger.warning("删除解析目录失败: %s", e)

    # Delete FAISS index files
    vector_dir = DATA_DIR / "vector_store"
    if vector_dir.exists():
        for idx_file in vector_dir.glob(f"*{doc_id}*"):
            try:
                idx_file.unlink()
            except Exception:
                pass

    await db.delete(doc)
    await db.commit()
    return {"message": "文档已删除，解析任务已取消"}


@router.post("/{doc_id}/reparse")
async def reparse_document(
    doc_id: int,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Re-trigger MinerU parsing for an existing document. Clears previous results."""
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")

    pdf_path = Path(doc.file_path)
    if not pdf_path.exists():
        raise HTTPException(400, "PDF 文件不存在，请重新上传")

    # Clear any previous cancellation flag
    _cancelled_docs.discard(doc_id)

    # Clear previous parsed data
    doc_dir = DATA_DIR / "parsed" / str(doc_id)
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
    doc_dir.mkdir(parents=True, exist_ok=True)

    # Also clear FAISS indices if any
    import os as _os
    for idx_file in doc_dir.parent.glob(f"faiss_*_{doc_id}.*"):
        try:
            idx_file.unlink()
        except Exception:
            pass

    doc.status = "pending"
    doc.progress = 0
    doc.total_pages = 0
    await db.commit()

    background_tasks.add_task(process_document_task, doc_id, pdf_path)
    logger.info("🔄 重新解析触发: doc_id=%d", doc_id)
    return {"message": "已触发重新解析", "doc_id": doc_id}


@router.get("/{doc_id}/subjects")
async def get_document_subjects(doc_id: int, db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user)):
    """Get subjects assigned to a document."""
    from app.models import Subject, DocumentSubject
    result = await db.execute(
        select(Subject).join(DocumentSubject).where(DocumentSubject.document_id == doc_id)
    )
    return [{"id": s.id, "name": s.name} for s in result.scalars().all()]


class SubjectIdsUpdate(BaseModel):
    subject_ids: list[int]


class DocTypeUpdate(BaseModel):
    doc_type: Optional[str] = None  # textbook | reference | None
    subject_id: int | None = None

class PrimaryDocUpdate(BaseModel):
    is_primary: bool

@router.put("/{doc_id}/type")
async def set_document_type(
    doc_id: int,
    data: DocTypeUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Set document type (textbook/reference) and optionally assign subject."""
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    if data.doc_type not in ("textbook", "reference", None):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "doc_type 必须为 textbook/reference/null")
    doc.doc_type = data.doc_type
    if data.subject_id:
        doc.subject_id = data.subject_id
        # Auto-create DocumentSubject association
        existing = await db.execute(
            select(DocumentSubject).where(
                DocumentSubject.document_id == doc_id,
                DocumentSubject.subject_id == data.subject_id
            )
        )
        if not existing.scalar_one_or_none():
            db.add(DocumentSubject(document_id=doc_id, subject_id=data.subject_id))
    await db.commit()
    logger.info("📘 文档 %d 类型设为 %s", doc_id, data.doc_type)
    return {"message": "文档类型已更新", "doc_type": data.doc_type}


@router.put("/{doc_id}/primary")
async def set_primary_textbook(
    doc_id: int,
    data: PrimaryDocUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Set document as primary textbook for its subject. Clears other primary markings."""
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")
    if data.is_primary and not doc.subject_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "请先为文档指定学科")
    if data.is_primary and doc.doc_type != "textbook":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "只有教材类型可设为主教材")

    if data.is_primary:
        # Clear existing primary docs for this subject
        await db.execute(select(Document).where(
            Document.subject_id == doc.subject_id, Document.is_primary == True
        ))
        docs = (await db.execute(select(Document).where(
            Document.subject_id == doc.subject_id, Document.is_primary == True
        ))).scalars().all()
        for d in docs:
            d.is_primary = False
        doc.is_primary = True
        # Also update Subject.primary_doc_id
        subj = (await db.execute(select(Subject).where(Subject.id == doc.subject_id))).scalar_one_or_none()
        if subj:
            subj.primary_doc_id = doc_id
    else:
        doc.is_primary = False
        if doc.subject_id:
            subj = (await db.execute(select(Subject).where(Subject.id == doc.subject_id))).scalar_one_or_none()
            if subj and subj.primary_doc_id == doc_id:
                subj.primary_doc_id = None
    await db.commit()
    return {"message": "主教材设置已更新", "is_primary": doc.is_primary}


@router.put("/{doc_id}/subjects")
async def update_document_subjects(
    doc_id: int,
    data: SubjectIdsUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Update subjects assigned to a document. Triggers knowledge tree building."""
    from sqlalchemy import delete as sqla_delete
    from app.services.knowledge_tree_service import KnowledgeTreeService
    from app.services.rag_service import RAGService

    # Check if document exists and is parsed
    doc_result = await db.execute(select(Document).where(Document.id == doc_id))
    doc = doc_result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "文档不存在")

    # Update subject associations
    await db.execute(sqla_delete(DocumentSubject).where(DocumentSubject.document_id == doc_id))
    for sid in data.subject_ids:
        db.add(DocumentSubject(document_id=doc_id, subject_id=sid))
    await db.commit()

    # Trigger knowledge tree building if subjects assigned and parsed
    result_msg = "学科分配已更新"
    if data.subject_ids and doc.status == "completed":
        kt_service = KnowledgeTreeService()
        rag = RAGService()

        for sid in data.subject_ids:
            try:
                # Check if this subject already has any KPs
                existing_kps = await db.execute(
                    select(KnowledgePoint).limit(1)
                )
                has_kps = existing_kps.scalar_one_or_none() is not None

                build_result = await kt_service.build_from_content_list(
                    doc_id=doc_id,
                    subject_id=sid,
                    db=db,
                    is_primary=not has_kps,
                )

                # Generate LLM summaries for newly created KPs (primary build only)
                if not has_kps and build_result.get("knowledge_points", 0) > 0:
                    try:
                        from app.services.llm_service import LLMService
                        llm = LLMService()
                        summary_count = await kt_service.generate_kp_summaries(db, llm_service=llm)
                        logger.info("Generated %d KP summaries for subject %d", len(summary_count), sid)
                    except Exception as e:
                        logger.warning("KP摘要生成跳过（LLM不可用）: %s", e)

                # Build/rebuild FAISS indices for this subject
                # FAISS built in background to avoid blocking
                logger.info("FAISS will be built in background for subject %d", sid)

                result_msg += f" | 学科{sid}: 构建{'主' if not has_kps else '补充'} {build_result['knowledge_points']}KP {build_result['content_chunks']}块 {build_result['exercises']}题"
            except Exception as e:
                logger.error("知识树构建失败 (doc=%d, subject=%d): %s", doc_id, sid, e)

    return {"message": result_msg, "subject_ids": data.subject_ids}

