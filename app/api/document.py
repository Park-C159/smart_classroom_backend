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
from app.models import Document, Subject, DocumentSubject, User, KnowledgePoint
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


def _kill_mineru_api_server():
    """终止常驻 mineru-api 服务（端口 8002），停止其 GPU 任务。"""
    import subprocess
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        for line in out.split("\n"):
            if ":8002" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                subprocess.run(["taskkill", "/F", "/PID", pid], capture_output=True)
                logger.info("已终止 mineru-api 服务 (PID %s)", pid)
                return
    except Exception as e:
        logger.warning("终止 mineru-api 失败: %s", e)


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
    _append_log(doc_id, "⏳ 开始 MinerU 解析（独立子进程懒加载模型，RAG 模型常驻 GPU 不受影响）...")
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
            _append_log(doc_id, "状态: parsed (33%) — 自动切块...")
            # 自动切块：按滑动窗口切成扁平分块（学科在构建时通过 document_subjects 归集）
            try:
                from app.models import DocumentSubject
                sids = (await db.execute(
                    select(DocumentSubject.subject_id).where(DocumentSubject.document_id == doc_id)
                )).scalars().all()
                sid = sids[0] if sids else None
                chunk_count = await RAGService._ingest_document_chunks(db, doc_id, sid)
                await db.commit()
                _append_log(doc_id, f"✅ 自动切块完成: {chunk_count} 块")
            except Exception as e:
                logger.warning("Auto chunking failed (non-fatal): %s", e)
                _append_log(doc_id, f"⚠️ 自动切块失败 (非致命): {e}")

            # 自动抽题：把解析结果里的题目抽取进独立试题库（各学科 in_qb 的书籍）
            try:
                from app.models import DocumentSubject
                from app.services.question_extractor import extract_questions_to_bank
                qb_rows = (await db.execute(
                    select(DocumentSubject).where(
                        DocumentSubject.document_id == doc_id,
                        DocumentSubject.in_qb.is_(True),
                    )
                )).scalars().all()
                total_q = 0
                for r in qb_rows:
                    res = await extract_questions_to_bank(db, doc_id, r.subject_id, chapter_override=r.qb_chapter)
                    total_q += res.get("extracted", 0)
                await db.commit()
                _append_log(doc_id, f"✅ 自动抽题完成: {total_q} 题")
            except Exception as e:
                logger.warning("Auto question extraction failed (non-fatal): %s", e)
                _append_log(doc_id, f"⚠️ 自动抽题失败 (非致命): {e}")

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

    # 流式写盘：分块读取，避免整文件读入内存（大文件更稳更快）
    with open(save_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB/块
            if not chunk:
                break
            f.write(chunk)

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
    query = select(Document).order_by(Document.id.asc())
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


@router.get("/library-membership")
async def get_library_membership(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """返回所有 (文档, 学科) 的知识库/题库归属，供向量库管理使用。"""
    from app.models import DocumentSubject
    rows = (await db.execute(select(DocumentSubject))).scalars().all()
    return [
        {"document_id": r.document_id, "subject_id": r.subject_id,
         "in_kb": bool(r.in_kb), "in_qb": bool(r.in_qb),
         "qb_chapter": r.qb_chapter}
        for r in rows
    ]


class LibraryMembershipUpdate(BaseModel):
    subject_id: int
    in_kb: bool = True
    in_qb: bool = True
    qb_chapter: str | None = None   # 抽题指定章（无结构纯题目列表时）


@router.put("/{doc_id}/library")
async def set_library_membership(
    doc_id: int,
    data: LibraryMembershipUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """设置某文档在某学科的知识库/题库归属；两者都 False 则移除。"""
    from app.models import DocumentSubject
    existing = await db.scalar(
        select(DocumentSubject).where(
            DocumentSubject.document_id == doc_id,
            DocumentSubject.subject_id == data.subject_id,
        )
    )
    if existing:
        if not data.in_kb and not data.in_qb:
            await db.delete(existing)
        else:
            existing.in_kb = data.in_kb
            existing.in_qb = data.in_qb
            existing.qb_chapter = data.qb_chapter
    elif data.in_kb or data.in_qb:
        db.add(DocumentSubject(
            document_id=doc_id, subject_id=data.subject_id,
            in_kb=data.in_kb, in_qb=data.in_qb, qb_chapter=data.qb_chapter,
        ))
    await db.commit()

    # 设为题库材料后，立即从解析结果抽取题目（题解分离 + 章/节对应）
    if data.in_qb:
        try:
            from app.services.question_extractor import extract_questions_to_bank
            res = await extract_questions_to_bank(db, doc_id, data.subject_id, chapter_override=data.qb_chapter)
            await db.commit()
            return {"message": f"已更新，抽取题目 {res.get('extracted', 0)} 道"}
        except Exception as e:
            logger.warning("设为题库后自动抽题失败 (doc=%d): %s", doc_id, e)

    return {"message": "已更新"}


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


@router.put("/{doc_id}/parsed/markdown")
async def update_parsed_markdown(doc_id: int, body: dict, db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin)):
    """保存解析校对后的 Markdown 内容（写回 result.json）。"""
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")
    markdown = body.get("markdown")
    if markdown is None:
        raise HTTPException(400, "缺少 markdown 字段")

    result_file = DATA_DIR / "parsed" / str(doc_id) / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "解析结果不存在")
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception:
        data = {}
    data["markdown"] = markdown
    result_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"message": "已保存", "doc_id": doc_id}


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

    from app.models import ContentChunk, DocumentSubject
    from sqlalchemy import delete as sa_delete

    # 1. 记录该文档所属学科（删除后需重建这些学科的知识库/题库）
    subj_rows = await db.execute(
        select(DocumentSubject.subject_id).where(DocumentSubject.document_id == doc_id)
    )
    subject_ids = [r[0] for r in subj_rows.all()]

    # 2. Signal any running background task to stop
    _cancelled_docs.add(doc_id)
    logger.info("🛑 取消文档 %d 的后台解析任务", doc_id)
    from app.services.document_processor import cancel_mineru_process
    cancel_mineru_process(doc_id)
    _kill_mineru_api_server()

    # 3. 删除该文档衍生的知识库/题库/习题数据（删除 = 废弃）
    await db.execute(sa_delete(ContentChunk).where(ContentChunk.source_doc_id == doc_id))
    await db.execute(sa_delete(DocumentSubject).where(DocumentSubject.document_id == doc_id))

    # 4. Delete original PDF + parsed output directory
    try:
        pdf_path = Path(doc.file_path)
        if pdf_path.exists():
            pdf_path.unlink()
    except Exception as e:
        logger.warning("删除 PDF 文件失败: %s", e)

    parsed_dir = DATA_DIR / "parsed" / str(doc_id)
    if parsed_dir.exists():
        try:
            shutil.rmtree(parsed_dir)
        except Exception as e:
            logger.warning("删除解析目录失败: %s", e)

    await db.delete(doc)
    await db.commit()

    # 5. 重建相关学科的知识库 + 题库索引（删除文档后内容已废弃）
    from app.services.rag_service import RAGService
    from app.services.gpu_manager import gpu_manager
    rag = RAGService()
    gpu_manager.to_gpu("embedding")  # 构建索引需要 embedding 在 GPU；clear_gpu 会把它挪到 CPU 导致跑满 CPU
    try:
        for sid in subject_ids:
            await rag.build_chunk_index(sid, db=db)
    finally:
        gpu_manager.restore_defaults()

    return {"message": "文档已删除，相关知识库/题库已重建", "rebuilt_subjects": subject_ids}


@router.post("/{doc_id}/cancel")
async def cancel_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """取消文档解析：停止后台任务，标记为 cancelled。"""
    doc = (await db.execute(select(Document).where(Document.id == doc_id))).scalar_one_or_none()
    if not doc:
        raise HTTPException(404, "文档不存在")
    _cancelled_docs.add(doc_id)
    from app.services.document_processor import cancel_mineru_process
    cancel_mineru_process(doc_id)
    _kill_mineru_api_server()
    doc.status = "cancelled"
    doc.progress = 0
    await db.commit()
    return {"message": "已取消解析", "doc_id": doc_id}


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

    # Trigger knowledge base (content chunks) building if subjects assigned and parsed
    result_msg = "学科分配已更新"
    if data.subject_ids and doc.status == "completed":
        kt_service = KnowledgeTreeService()

        for sid in data.subject_ids:
            try:
                # 知识树已改为手动构建，这里只生成内容块（知识库，供答疑检索）
                build_result = await kt_service.build_from_content_list(
                    doc_id=doc_id,
                    subject_id=sid,
                    db=db,
                    is_primary=True,
                )

                result_msg += f" | 学科{sid}: {build_result['content_chunks']}块"
            except Exception as e:
                logger.error("知识库构建失败 (doc=%d, subject=%d): %s", doc_id, sid, e)

    return {"message": result_msg, "subject_ids": data.subject_ids}

