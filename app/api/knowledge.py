"""Knowledge Tree API — CRUD for KPs, content chunks, and tree building."""
import logging

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import func, select, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.security import get_current_user, get_admin_user, get_teacher_or_admin
from app.models import (
    KnowledgePoint, ContentChunk, Exercise, Document,
)
from app.services.knowledge_tree_service import KnowledgeTreeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["知识树"])
tree_service = KnowledgeTreeService()


# ── Pydantic schemas ──

class KPCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=20)
    title: str = Field(..., min_length=1, max_length=200)
    summary: str = ""
    parent_id: str | None = None
    chapter: str | None = None
    level: int = 2
    sort_order: int = 0


class KPUpdate(BaseModel):
    title: str | None = None
    summary: str | None = None
    parent_id: str | None = None
    chapter: str | None = None
    sort_order: int | None = None


class ChunkCreate(BaseModel):
    kp_id: str
    chunk_type: str = "definition"
    content: str = Field(..., min_length=1)
    page_number: int | None = None


class ChunkUpdate(BaseModel):
    content: str | None = None
    chunk_type: str | None = None
    page_number: int | None = None


# ── Tree ──

# Section titles that are exercise containers, not real knowledge sections
_EXERCISE_TITLES = {"习题", "补充题"}

@router.get("/tree")
async def get_knowledge_tree(db: AsyncSession = Depends(get_db)):
    """Get knowledge tree — textbook KPs only, no reference doc, no exercise sections."""
    result = await db.execute(
        select(KnowledgePoint).order_by(KnowledgePoint.sort_order)
    )
    kps = result.scalars().all()

    # 1) Exclude reference-document KPs (D2-* prefix = doc 2 = reference)
    kps = [kp for kp in kps if not kp.id.startswith("D2-")]

    # 2) Deduplicate: prefer original KP-N over D1-KP-N (same textbook rebuilt)
    # Collect all IDs from duplicate chapter trees to skip
    seen_chapters = set()
    dup_chapter_ids = set()  # chapter IDs to skip (D1-KP-N duplicates)
    for kp in kps:
        if kp.level == 0:
            ch_num = kp.id.rsplit("-", 1)[-1]
            if ch_num in seen_chapters:
                dup_chapter_ids.add(kp.id)
            else:
                seen_chapters.add(ch_num)
    # Also skip all descendants of duplicate chapters
    if dup_chapter_ids:
        dup_descendants = set()
        id_to_parent = {kp.id: kp.parent_id for kp in kps}
        for kp in kps:
            pid = kp.parent_id
            while pid:
                if pid in dup_chapter_ids:
                    dup_descendants.add(kp.id)
                    break
                pid = id_to_parent.get(pid)
        dup_chapter_ids |= dup_descendants
    kps = [kp for kp in kps if kp.id not in dup_chapter_ids]

    # 2) Exclude exercise/supplement sections (already in question bank)
    def _is_exercise_section(title: str) -> bool:
        t = title.strip()
        return t in _EXERCISE_TITLES or "习题" in t or "补充题" in t
    exercise_parents = {kp.id for kp in kps if kp.level == 1 and _is_exercise_section(kp.title)}
    skip_ids = set(exercise_parents)
    id_to_parent = {kp.id: kp.parent_id for kp in kps}
    for kp in kps:
        pid = kp.parent_id
        while pid:
            if pid in exercise_parents:
                skip_ids.add(kp.id)
                break
            pid = id_to_parent.get(pid)
    kps = [kp for kp in kps if kp.id not in skip_ids]

    # Build tree structure (recursive, supports arbitrary depth)
    kp_map = {kp.id: kp for kp in kps}
    children_map = {}  # parent_id → list of children
    for kp in kps:
        if kp.parent_id:
            children_map.setdefault(kp.parent_id, []).append(kp)

    def build_node(kp):
        node = _kp_to_node(kp)
        kids = children_map.get(kp.id, [])
        if kids:
            node["children"] = [build_node(c) for c in kids]
        return node

    root_nodes = [kp for kp in kps if not kp.parent_id]
    tree = [build_node(kp) for kp in root_nodes]

    # Any orphaned KPs (parent_id points to nonexistent node)
    all_ids = set(kp_map.keys())
    orphans = [kp for kp in kps if kp.parent_id and kp.parent_id not in all_ids]
    for kp in orphans:
        tree.append(_kp_to_node(kp))

    return {"data": tree, "total": len(kps)}


def _kp_to_node(kp: KnowledgePoint) -> dict:
    return {
        "id": kp.id,
        "title": kp.title,
        "summary": kp.summary,
        "parent_id": kp.parent_id,
        "chapter": kp.chapter,
        "level": kp.level,
        "sort_order": kp.sort_order,
        "created_at": kp.created_at.isoformat() if kp.created_at else None,
    }


# ── Single KP ──

@router.get("/kp/{kp_id}")
async def get_kp(kp_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single knowledge point with its chunks and exercises."""
    result = await db.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.id == kp_id)
        .options(selectinload(KnowledgePoint.content_chunks), selectinload(KnowledgePoint.exercises))
    )
    kp = result.scalar_one_or_none()
    if not kp:
        raise HTTPException(404, "知识点不存在")

    return {
        "kp": _kp_to_node(kp),
        "chunks": [
            {"id": c.id, "chunk_type": c.chunk_type, "content": c.content,
             "page_number": c.page_number}
            for c in (kp.content_chunks or [])
        ],
        "exercises": [
            {"id": e.id, "question_text": e.question_text, "answer_text": e.answer_text,
             "question_type": e.question_type, "difficulty": e.difficulty, "source": e.source}
            for e in (kp.exercises or [])
        ],
    }


# ── Create KP ──

@router.post("/kp")
async def create_kp(
    data: KPCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Create a new knowledge point."""
    existing = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == data.id))
    if existing:
        raise HTTPException(409, "知识点 ID 已存在")

    kp = KnowledgePoint(
        id=data.id, title=data.title, summary=data.summary,
        parent_id=data.parent_id, chapter=data.chapter,
        level=data.level, sort_order=data.sort_order,
    )
    db.add(kp)
    await db.flush()
    return _kp_to_node(kp)


# ── Update KP ──

@router.put("/kp/{kp_id}")
async def update_kp(
    kp_id: str,
    data: KPUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Update a knowledge point (supports partial update)."""
    kp = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == kp_id))
    if not kp:
        raise HTTPException(404, "知识点不存在")

    if data.title is not None:
        kp.title = data.title
    if data.summary is not None:
        kp.summary = data.summary
    if data.parent_id is not None:
        kp.parent_id = data.parent_id
    if data.chapter is not None:
        kp.chapter = data.chapter
    if data.sort_order is not None:
        kp.sort_order = data.sort_order

    await db.flush()
    return _kp_to_node(kp)


# ── Delete KP ──

@router.delete("/kp/{kp_id}")
async def delete_kp(
    kp_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Delete a knowledge point and its associated chunks/exercises."""
    kp = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == kp_id))
    if not kp:
        raise HTTPException(404, "知识点不存在")

    # Cascade delete children
    children = (await db.execute(
        select(KnowledgePoint).where(KnowledgePoint.parent_id == kp_id)
    )).scalars().all()
    for child in children:
        await db.delete(child)

    await db.delete(kp)
    return {"message": f"知识点 {kp_id} 已删除"}


# ── Bulk reorder ──

@router.put("/reorder")
async def reorder_kps(
    orders: list[dict],  # [{id: "KP-1.1", sort_order: 0}, ...]
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Bulk update sort_order for knowledge points."""
    for item in orders:
        kp = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == item["id"]))
        if kp:
            kp.sort_order = item.get("sort_order", 0)
            if "parent_id" in item:
                kp.parent_id = item["parent_id"]
    await db.flush()
    return {"message": f"已更新 {len(orders)} 个知识点的排序"}


# ── Content chunks ──

@router.get("/chunks/{kp_id}")
async def list_chunks(kp_id: str, db: AsyncSession = Depends(get_db)):
    """List content chunks for a knowledge point."""
    result = await db.execute(
        select(ContentChunk).where(ContentChunk.kp_id == kp_id).order_by(ContentChunk.id)
    )
    chunks = result.scalars().all()
    return {
        "data": [
            {"id": c.id, "kp_id": c.kp_id, "chunk_type": c.chunk_type,
             "content": c.content, "page_number": c.page_number}
            for c in chunks
        ],
        "total": len(chunks),
    }


@router.post("/chunks")
async def create_chunk(
    data: ChunkCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Add a content chunk to a knowledge point."""
    chunk = ContentChunk(
        kp_id=data.kp_id, chunk_type=data.chunk_type,
        content=data.content, page_number=data.page_number,
    )
    db.add(chunk)
    await db.flush()
    return {"id": chunk.id, "kp_id": chunk.kp_id, "chunk_type": chunk.chunk_type,
            "content": chunk.content, "page_number": chunk.page_number}


@router.put("/chunks/{chunk_id}")
async def update_chunk(
    chunk_id: int,
    data: ChunkUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Update a content chunk."""
    chunk = await db.scalar(select(ContentChunk).where(ContentChunk.id == chunk_id))
    if not chunk:
        raise HTTPException(404, "内容块不存在")
    if data.content is not None:
        chunk.content = data.content
    if data.chunk_type is not None:
        chunk.chunk_type = data.chunk_type
    if data.page_number is not None:
        chunk.page_number = data.page_number
    await db.flush()
    return {"id": chunk.id, "kp_id": chunk.kp_id, "chunk_type": chunk.chunk_type,
            "content": chunk.content, "page_number": chunk.page_number}


@router.delete("/chunks/{chunk_id}")
async def delete_chunk(
    chunk_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Delete a content chunk."""
    chunk = await db.scalar(select(ContentChunk).where(ContentChunk.id == chunk_id))
    if not chunk:
        raise HTTPException(404, "内容块不存在")
    await db.delete(chunk)
    return {"message": "内容块已删除"}


# ── Chunk management (by subject/section) ──

class ChunkUpdate(BaseModel):
    content: str | None = None
    chunk_type: str | None = None
    page_number: int | None = None

@router.get("/chunks")
async def list_chunks(
    subject_id: int | None = Query(None, description="Subject ID"),
    section_id: str | None = Query(None, description="Section KP ID, e.g. KP-1.1"),
    chunk_type: str | None = Query(None, description="Filter by type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, le=2000, description="Page size"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """List content chunks with filters for management view."""
    from sqlalchemy import func as sa_func

    # Build section filter: match kp_id prefix (e.g., "KP-1.1" matches "KP-1.1.1", "KP-1.1.2", ...)
    if section_id:
        prefix = section_id + "."
        chunks = await db.execute(
            select(ContentChunk).where(
                ContentChunk.kp_id.like(prefix + "%"),
                ContentChunk.subject_id == subject_id if subject_id else True,
                ContentChunk.chunk_type == chunk_type if chunk_type else True,
            ).order_by(ContentChunk.id).limit(page_size).offset((page - 1) * page_size)
        )
    else:
        chunks = await db.execute(
            select(ContentChunk).where(
                ContentChunk.subject_id == subject_id if subject_id else True,
                ContentChunk.chunk_type == chunk_type if chunk_type else True,
            ).order_by(ContentChunk.id).limit(page_size).offset((page - 1) * page_size)
        )
    return [{
        "id": c.id, "kp_id": c.kp_id, "chunk_type": c.chunk_type,
        "content": c.content[:300] + ("..." if len(c.content or "") > 300 else ""),
        "full_content": c.content, "page_number": c.page_number,
        "source_doc_id": c.source_doc_id, "images": c.images,
    } for c in chunks.scalars().all()]


@router.put("/chunks/{chunk_id}")
async def update_chunk(
    chunk_id: int,
    data: ChunkUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Update a content chunk (edit content, type, page number)."""
    chunk = await db.scalar(select(ContentChunk).where(ContentChunk.id == chunk_id))
    if not chunk:
        raise HTTPException(404, "内容块不存在")

    if data.content is not None:
        chunk.content = data.content[:5000]
    if data.chunk_type is not None:
        chunk.chunk_type = data.chunk_type
    if data.page_number is not None:
        chunk.page_number = data.page_number

    await db.commit()
    return {"message": "分块已更新", "id": chunk_id}


@router.post("/chunks/rebuild-index")
async def rebuild_kb_index(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Rebuild knowledge base FAISS index for a subject."""
    from app.services.rag_service import RAGService
    from app.services.gpu_manager import gpu_manager

    gpu_manager.clear_gpu()
    try:
        rag = RAGService()
        result = await rag.build_kb_index(subject_id, db=db)
        return {"message": "知识库索引已重建", "result": result}
    finally:
        gpu_manager.restore_defaults()


# ── Question Bank management ──

@router.get("/question-bank")
async def list_question_bank(
    subject_id: int | None = Query(None),
    chapter: str | None = Query(None),
    source: str | None = Query(None),
    source_doc_id: int | None = Query(None, description="Filter by source document ID"),
    question_type: str | None = Query(None, description="Filter by question type"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, le=2000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """List question bank entries."""
    from app.models import QuestionBank
    conditions = []
    if subject_id:
        conditions.append(QuestionBank.subject_id == subject_id)
    if chapter:
        conditions.append(QuestionBank.chapter == chapter)
    if source:
        conditions.append(QuestionBank.source == source)
    if source_doc_id:
        conditions.append(QuestionBank.source_doc_id == source_doc_id)
    if question_type:
        conditions.append(QuestionBank.question_type == question_type)

    q = select(QuestionBank).where(*conditions).order_by(QuestionBank.id).limit(page_size).offset((page - 1) * page_size)
    result = await db.execute(q)
    questions = result.scalars().all()
    return [{
        "id": q.id, "question_text": q.question_text,
        "answer_text": q.answer_text, "question_type": q.question_type,
        "difficulty": q.difficulty, "source": q.source,
        "source_doc_id": q.source_doc_id, "page_number": q.page_number,
        "chapter": q.chapter, "kp_id": q.kp_id,
        "images": q.images, "verified": q.verified,
        "merged_from": q.merged_from,
    } for q in questions]


@router.put("/question-bank/{question_id}")
async def update_question_bank(
    question_id: int,
    data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Edit a question bank entry."""
    from app.models import QuestionBank
    q = await db.get(QuestionBank, question_id)
    if not q:
        raise HTTPException(404, "题目不存在")
    for field in ("question_text", "answer_text", "question_type", "chapter",
                  "page_number", "source_doc_id", "difficulty", "verified"):
        if field in data:
            setattr(q, field, data[field])
    # Update embedding text when question_text changes
    if "question_text" in data:
        q.embedding_text = data["question_text"]
    await db.commit()
    return {"message": "已更新", "id": question_id}


@router.post("/question-bank")
async def create_question_bank(
    data: dict,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Add a new question to the question bank (teacher/admin)."""
    from app.models import QuestionBank
    question_text = (data.get("question_text") or "").strip()
    if not question_text:
        raise HTTPException(400, "题目内容不能为空")
    q = QuestionBank(
        question_text=question_text,
        answer_text=data.get("answer_text") or "",
        question_type=data.get("question_type") or "reference",
        difficulty=data.get("difficulty", 3),
        source=data.get("source") or "manual",
        source_doc_id=data.get("source_doc_id"),
        page_number=data.get("page_number"),
        chapter=data.get("chapter"),
        kp_id=data.get("kp_id"),
        subject_id=data.get("subject_id", 9),
        embedding_text=question_text,
        verified=data.get("verified", False),
    )
    db.add(q)
    await db.commit()
    await db.refresh(q)
    return {"id": q.id, "message": "题目已添加，重建索引后即可用于检索"}


@router.post("/question-bank/rebuild-index")
async def rebuild_qb_index(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Rebuild question bank FAISS index for a subject."""
    from app.services.rag_service import RAGService
    from app.services.gpu_manager import gpu_manager

    gpu_manager.clear_gpu()
    try:
        rag = RAGService()
        result = await rag.build_qb_index(subject_id, db=db)
        return {"message": "题库索引已重建", "result": result}
    finally:
        gpu_manager.restore_defaults()


# ── Build tree from parsed document ──

@router.post("/build-from-doc/{doc_id}")
async def build_tree_from_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Build knowledge tree from a parsed document using content_list_v2.json."""
    # Find the subject this document is assigned to
    from sqlalchemy import text as sa_text
    subj_result = await db.execute(
        sa_text("SELECT subject_id FROM document_subjects WHERE document_id = :did LIMIT 1"),
        {"did": doc_id}
    )
    subj_row = subj_result.fetchone()
    subject_id = subj_row[0] if subj_row else None

    if not subject_id:
        raise HTTPException(400, "请先将文档分配到学科后再构建知识树")

    # Check if this is the primary document for the subject
    existing_kps = await db.scalar(
        select(func.count()).select_from(KnowledgePoint).where(KnowledgePoint.id.like("KP-%"))
    )
    is_primary = existing_kps == 0

    # Build from content_list_v2.json
    result = await tree_service.build_from_content_list(doc_id, subject_id, db, is_primary=is_primary)

    # Preview structure
    content_list_path = tree_service._find_content_list(doc_id)
    structure = {"chapters": [], "exercises": []}
    if content_list_path:
        import json as _json
        with open(content_list_path, "r", encoding="utf-8") as f:
            pages = _json.load(f)
        structure = tree_service._parse_structure(pages)

    return {
        "doc_id": doc_id,
        "subject_id": subject_id,
        "result": result,
        "preview": {
            "chapters": len(structure["chapters"]),
            "chapter_titles": [c["title"] for c in structure["chapters"]],
            "sections": sum(len(c.get("sections", [])) for c in structure["chapters"]),
            "exercises_found": len(structure.get("exercises", [])),
        },
    }


# ── Preview structure (no DB write) ──

@router.get("/preview/{doc_id}")
async def preview_tree_structure(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Preview the knowledge tree structure extracted from a parsed document (no DB write)."""
    from app.services.document_processor import DocumentProcessor
    processor = DocumentProcessor()

    parsed = processor.get_parsed_result(doc_id)
    if not parsed:
        raise HTTPException(404, "文档尚未解析完成")

    markdown = parsed.get("markdown", "")
    if not markdown:
        raise HTTPException(400, "解析结果中没有 Markdown 内容")

    structure = tree_service.preview_structure(markdown)
    return structure


# ── Stats ──

@router.get("/stats")
async def get_tree_stats(db: AsyncSession = Depends(get_db)):
    """Get knowledge tree statistics."""
    total_kps = await db.scalar(select(func.count(KnowledgePoint.id)))
    total_chunks = await db.scalar(select(func.count(ContentChunk.id)))
    total_ex = await db.scalar(select(func.count(Exercise.id)))
    chapters = (await db.execute(
        select(func.distinct(KnowledgePoint.chapter))
        .where(KnowledgePoint.chapter.isnot(None))
    )).all()

    return {
        "knowledge_points": total_kps or 0,
        "content_chunks": total_chunks or 0,
        "exercises": total_ex or 0,
        "chapters": len(chapters) if chapters else 0,
    }


# ── KP Review (admin) ──

@router.get("/review")
async def review_knowledge_tree(
    chapter: str | None = None,
    has_summary: bool | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Get all KPs with summary, chunk count, and exercise count for review."""
    result = await db.execute(
        select(KnowledgePoint).order_by(KnowledgePoint.sort_order)
    )
    kps = result.scalars().all()

    # Collect chunk counts per KP
    from sqlalchemy import func as sa_func
    chunk_counts = {}
    if kps:
        kp_ids = [kp.id for kp in kps]
        cresult = await db.execute(
            select(ContentChunk.kp_id, sa_func.count(ContentChunk.id))
            .where(ContentChunk.kp_id.in_(kp_ids))
            .group_by(ContentChunk.kp_id)
        )
        chunk_counts = {row[0]: row[1] for row in cresult.all()}

    ex_counts = {}
    if kps:
        eresult = await db.execute(
            select(Exercise.kp_id, sa_func.count(Exercise.id))
            .where(Exercise.kp_id.in_(kp_ids))
            .group_by(Exercise.kp_id)
        )
        ex_counts = {row[0]: row[1] for row in eresult.all()}

    # Get document titles for exercises
    doc_result = await db.execute(select(Document.id, Document.title))
    doc_titles = {row[0]: row[1] for row in doc_result.all()}

    # Build sections with KPs
    kp_map = {kp.id: kp for kp in kps}
    children_map = {}
    for kp in kps:
        if kp.parent_id:
            children_map.setdefault(kp.parent_id, []).append(kp)

    items = []
    for kp in kps:
        if kp.level != 2:
            continue  # Only review level-2 KPs
        if chapter and not kp.id.startswith(f"KP-{chapter}") and not kp.id.startswith(chapter):
            continue
        if has_summary is True and (not kp.summary or kp.summary.strip() == ''):
            continue
        if has_summary is False and kp.summary and kp.summary.strip() != '':
            continue

        # Build path
        path = []
        pid = kp.parent_id
        while pid and pid in kp_map:
            path.insert(0, kp_map[pid].title)
            pid = kp_map[pid].parent_id
        path.insert(0, kp_map[kp.id.split('.')[0]].title if '.' in kp.id else '')

        # Get exercises for the parent section
        sec_id = '.'.join(kp.id.split('.')[:2])
        sec_exs = await db.execute(
            select(Exercise).where(Exercise.kp_id.like(f"{sec_id}.%"))
        )
        exercises = sec_exs.scalars().all()
        ex_list = [
            {
                "id": e.id, "question_text": e.question_text[:120],
                "answer_text": e.answer_text[:80] if e.answer_text else None,
                "question_type": e.question_type, "difficulty": e.difficulty,
                "page_number": e.page_number,
                "source_doc_title": doc_titles.get(e.source_doc_id, "") if e.source_doc_id else "",
            }
            for e in exercises
        ]

        items.append({
            "id": kp.id,
            "title": kp.title,
            "summary": kp.summary or "",
            "chapter_path": " > ".join(path),
            "chunk_count": chunk_counts.get(kp.id, 0),
            "exercise_count": ex_counts.get(kp.id, 0),
            "section_exercises": ex_list,
            "level": kp.level,
            "sort_order": kp.sort_order,
        })

    return {"data": items, "total": len(items)}


@router.post("/summarize/{kp_id}")
async def summarize_single_kp(
    kp_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Re-summarize a single KP with full content context."""
    from app.services.llm_service import LLMService
    kp = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == kp_id))
    if not kp:
        raise HTTPException(404, "知识点不存在")

    chunks = (await db.execute(
        select(ContentChunk).where(ContentChunk.kp_id == kp_id).order_by(ContentChunk.id)
    )).scalars().all()
    context = "\n".join(c.content[:500] for c in chunks) if chunks else kp.title

    try:
        llm = LLMService()
        summary = ""
        for _, text in llm.get_stream_response(
            query=f"请根据以下教材内容，提炼该知识点的核心概念名称（8-15字），要求保留LaTeX数学公式：\n{context}",
            context=None,
            system_prompt="你是一名数学教材编辑，请提炼精确的知识点名称。要求：1.保留LaTeX公式 2.用准确术语 3.只返回名称，不要解释",
        ):
            summary += text
        kp.summary = summary.strip().split('\n')[0][:60]
        await db.flush()
        return {"kp_id": kp_id, "summary": kp.summary, "chunks_used": len(chunks)}
    except Exception as e:
        raise HTTPException(500, f"摘要生成失败: {str(e)}")


# ── Legacy Summarization ──

@router.post("/summarize")
async def summarize_kps(
    force: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Generate LLM summaries for all level-2 knowledge points."""
    from app.services.llm_service import LLMService
    try:
        llm = LLMService()
        results = await tree_service.generate_kp_summaries(db, llm_service=llm, force=force)
        return {"message": f"已生成 {len(results)} 个知识点摘要", "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"摘要生成失败: {str(e)}")
