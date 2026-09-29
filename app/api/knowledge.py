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
    KnowledgePoint, ContentChunk, Document, TestQuestion,
)
from app.services.knowledge_tree_service import KnowledgeTreeService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/knowledge", tags=["知识树"])
tree_service = KnowledgeTreeService()


@router.get("/build-status")
async def get_build_status(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """返回各学科知识库构建状态（是否已构建、分块数）。"""
    import json as _json
    from pathlib import Path as _Path
    from app.config import settings as _settings
    from app.models import Subject
    subjects = (await db.execute(
        select(Subject).where(Subject.is_active == True).order_by(Subject.id)
    )).scalars().all()
    result = []
    for s in subjects:
        store = _Path(_settings.DATA_DIR) / "vector_store" / str(s.id)
        chunks = 0
        meta = store / "chunk_meta.json"
        if meta.exists():
            try:
                chunks = len(_json.loads(meta.read_text(encoding="utf-8")))
            except Exception:
                chunks = 0
        result.append({
            "subject_id": s.id,
            "subject_name": s.name,
            "kb_built": (store / "faiss_chunk.index").exists(),
            "kb_chunks": chunks,
        })
    return result


# ── Pydantic schemas ──

class KPCreate(BaseModel):
    id: str = Field(..., min_length=1, max_length=20)
    title: str = Field(..., min_length=1, max_length=200)
    summary: str = ""
    parent_id: str | None = None
    chapter: str | None = None
    level: int = 2
    sort_order: int = 0
    subject_id: int = Field(..., description="所属学科")


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
async def get_knowledge_tree(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
):
    """Get knowledge tree for a subject (manually built by admin/teacher)."""
    result = await db.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.subject_id == subject_id)
        .order_by(KnowledgePoint.sort_order, KnowledgePoint.id)
    )
    kps = result.scalars().all()

    # Build tree structure (recursive, supports arbitrary depth)
    kp_map = {kp.id: kp for kp in kps}
    children_map: dict[str, list[KnowledgePoint]] = {}
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
        "subject_id": kp.subject_id,
        "created_at": kp.created_at.isoformat() if kp.created_at else None,
    }


# ── Single KP ──

@router.get("/kp/{kp_id}")
async def get_kp(kp_id: str, db: AsyncSession = Depends(get_db)):
    """Get a single knowledge point with its chunks and exercises."""
    result = await db.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.id == kp_id)
        .options(selectinload(KnowledgePoint.content_chunks))
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
        subject_id=data.subject_id,
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
    """删除知识点及其所有子节点（章→节）。

    用原生 SQL 删除：ORM 的 db.delete() 会对 content_chunks/exercises 等关系
    尝试把 kp_id 置空，而这些列是 NOT NULL，导致 IntegrityError。
    """
    from sqlalchemy import text as sa_text
    node = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == kp_id))
    if not node:
        raise HTTPException(404, "知识点不存在")
    subject_id = node.subject_id
    is_chapter = node.level == 0

    # 收集该节点及全部后代 id（层级浅，用 BFS）
    ids = [kp_id]
    frontier = [kp_id]
    while frontier:
        rows = (await db.execute(
            select(KnowledgePoint.id).where(KnowledgePoint.parent_id.in_(frontier))
        )).scalars().all()
        frontier = [r for r in rows if r not in ids]
        ids.extend(frontier)

    placeholders = ",".join(f":id{i}" for i in range(len(ids)))
    params = {f"id{i}": v for i, v in enumerate(ids)}
    await db.execute(
        sa_text(f"DELETE FROM knowledge_points WHERE id IN ({placeholders})"), params
    )
    await db.commit()

    # 删除章后，其下分块成了孤儿，只对这些块按内容相似度重新匹配到剩余章，并重写索引元数据
    if is_chapter and subject_id:
        try:
            from sqlalchemy import or_
            from app.services.rag_service import RAGService
            from app.models import ChunkKpMap
            orphan_ids = list((await db.execute(
                select(ChunkKpMap.chunk_id).where(
                    or_(ChunkKpMap.chapter_id.in_(ids), ChunkKpMap.section_id.in_(ids))
                )
            )).scalars().all())
            if orphan_ids:
                rag = RAGService()
                await rag.match_chunks_to_chapters(db, subject_id, chunk_ids=orphan_ids)
                await rag._write_chunk_meta(db, subject_id)
        except Exception as e:
            logger.warning("删除章后重新匹配分块失败: %s", e)

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


@router.post("/renumber")
async def renumber_subject(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """按当前 sort_order 重新给章/节编号：id 重写为 KP-{sid}-{n} / KP-{sid}-{n}.{m}，并级联更新引用。"""
    from sqlalchemy import text as sa_text

    nodes = (await db.execute(
        select(KnowledgePoint).where(KnowledgePoint.subject_id == subject_id)
    )).scalars().all()

    # 章（level 0）按 sort_order 排；节挂在其父章下按 sort_order 排
    chapters = sorted([n for n in nodes if n.level == 0], key=lambda n: (n.sort_order, n.id))
    old_to_new: dict[str, str] = {}
    for i, ch in enumerate(chapters, 1):
        new_ch = f"KP-{subject_id}-{i}"
        old_to_new[ch.id] = new_ch
        secs = sorted([n for n in nodes if n.parent_id == ch.id], key=lambda n: (n.sort_order, n.id))
        for j, sec in enumerate(secs, 1):
            old_to_new[sec.id] = f"{new_ch}.{j}"

    if not old_to_new:
        return {"message": "该学科暂无节点", "count": 0}

    # 两阶段改名（id 与 parent_id 都走临时值），避免主键冲突，也避免章交换时 parent_id 串行
    old_to_tmp = {old: f"__tmp_{subject_id}_{k}" for k, old in enumerate(old_to_new)}
    # 1) id: old → tmp
    for old, tmp in old_to_tmp.items():
        await db.execute(sa_text("UPDATE knowledge_points SET id=:n WHERE id=:o"), {"n": tmp, "o": old})
    # 2) parent_id: old → tmp（先跟着临时值，防止新旧 id 交换时二次覆盖）
    for old, tmp in old_to_tmp.items():
        await db.execute(sa_text("UPDATE knowledge_points SET parent_id=:n WHERE parent_id=:o"), {"n": tmp, "o": old})
    # 3) id: tmp → new
    for old, new in old_to_new.items():
        await db.execute(sa_text("UPDATE knowledge_points SET id=:n WHERE id=:o"), {"n": new, "o": old_to_tmp[old]})
    # 4) parent_id: tmp → new
    for old, new in old_to_new.items():
        await db.execute(sa_text("UPDATE knowledge_points SET parent_id=:n WHERE parent_id=:o"), {"n": new, "o": old_to_tmp[old]})

    # 级联更新引用该树节点的其它表（单列字符串/外键）
    for table, col in (
        ("content_chunks", "kp_id"),
        ("kp_mastery", "kp_id"),
        ("discussions", "kp_id"),
        ("test_questions", "kp_id"),
        ("paper_questions", "kp_id"),
        ("chunk_kp_map", "chapter_id"),
        ("chunk_kp_map", "section_id"),
    ):
        for old, new in old_to_new.items():
            await db.execute(
                sa_text(f"UPDATE {table} SET {col}=:n WHERE {col}=:o"), {"n": new, "o": old}
            )

    await db.commit()
    return {"message": f"已重新编号 {len(old_to_new)} 个节点", "count": len(old_to_new), "mapping": old_to_new}


@router.post("/match-chunks")
async def match_chunks_endpoint(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """手动触发：把该学科知识库分块按内容相似度重新匹配到知识树的章并重建索引。"""
    from app.services.rag_service import RAGService
    return await RAGService().build_chunk_index(subject_id, db=db)


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
    """Add a content chunk to a knowledge point (subject inherited from KP)."""
    kp = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == data.kp_id))
    chunk = ContentChunk(
        kp_id=data.kp_id, chunk_type=data.chunk_type,
        content=data.content, page_number=data.page_number,
        subject_id=kp.subject_id if kp else None,
    )
    db.add(chunk)
    await db.flush()
    return {"id": chunk.id, "kp_id": chunk.kp_id, "chunk_type": chunk.chunk_type,
            "content": chunk.content, "page_number": chunk.page_number,
            "subject_id": chunk.subject_id}


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


class ChunkChapterUpdate(BaseModel):
    chapter_id: str | None = None
    section_id: str | None = None


@router.get("/chunks")
async def list_chunks(
    subject_id: int | None = Query(None, description="Subject ID"),
    chapter_id: str | None = Query(None, description="章 ID"),
    section_id: str | None = Query(None, description="节 ID"),
    chunk_type: str | None = Query(None, description="Filter by type"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, le=2000, description="Page size"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """List content chunks with chapter/section mapping for management view."""
    from app.models import ChunkKpMap

    conds = []
    if subject_id:
        conds.append(ContentChunk.subject_id == subject_id)
    if chunk_type:
        conds.append(ContentChunk.chunk_type == chunk_type)

    # 章节过滤走 chunk_kp_map
    if chapter_id or section_id:
        mconds = []
        if chapter_id:
            mconds.append(ChunkKpMap.chapter_id == chapter_id)
        if section_id:
            mconds.append(ChunkKpMap.section_id == section_id)
        rows = await db.execute(select(ChunkKpMap.chunk_id).where(and_(*mconds)))
        cids = [r[0] for r in rows.all()]
        if not cids:
            return []
        conds.append(ContentChunk.id.in_(cids))

    chunks = (await db.execute(
        select(ContentChunk).where(and_(*conds)).order_by(ContentChunk.id)
        .offset((page - 1) * page_size).limit(page_size)
    )).scalars().all()

    chunk_ids = [c.id for c in chunks]
    map_rows = []
    if chunk_ids:
        map_rows = (await db.execute(
            select(ChunkKpMap).where(ChunkKpMap.chunk_id.in_(chunk_ids))
        )).scalars().all()
    m_by = {m.chunk_id: m for m in map_rows}

    kp_ids = set()
    for m in map_rows:
        if m.chapter_id:
            kp_ids.add(m.chapter_id)
        if m.section_id:
            kp_ids.add(m.section_id)
    kp_map = {}
    if kp_ids:
        kps = (await db.execute(select(KnowledgePoint).where(KnowledgePoint.id.in_(kp_ids)))).scalars().all()
        kp_map = {k.id: k for k in kps}

    out = []
    for c in chunks:
        m = m_by.get(c.id)
        chapter_id = m.chapter_id if m else None
        section_id = m.section_id if m else None
        out.append({
            "id": c.id, "kp_id": c.kp_id, "chunk_type": c.chunk_type,
            "chapter_id": chapter_id, "section_id": section_id,
            "chapter_title": kp_map[chapter_id].title if chapter_id in kp_map else "",
            "section_title": kp_map[section_id].title if section_id in kp_map else "",
            "content": c.content[:300] + ("..." if len(c.content or "") > 300 else ""),
            "full_content": c.content, "page_number": c.page_number,
            "source_doc_id": c.source_doc_id, "images": c.images,
        })
    return out


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


@router.put("/chunks/{chunk_id}/chapter")
async def update_chunk_chapter(
    chunk_id: int,
    data: ChunkChapterUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """手动修改分块归属的章/节（覆盖自动匹配）。"""
    from app.models import ChunkKpMap

    chunk = await db.scalar(select(ContentChunk).where(ContentChunk.id == chunk_id))
    if not chunk:
        raise HTTPException(404, "内容块不存在")

    chapter_id = data.chapter_id
    section_id = data.section_id
    if section_id and not chapter_id:
        sec = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == section_id))
        if sec:
            chapter_id = sec.parent_id

    row = await db.get(ChunkKpMap, chunk_id)
    if row:
        row.chapter_id = chapter_id
        row.section_id = section_id
    else:
        db.add(ChunkKpMap(
            chunk_id=chunk_id, subject_id=chunk.subject_id,
            chapter_id=chapter_id, section_id=section_id, similarity=1.0,
        ))
    chunk.kp_id = chapter_id or "flat"
    await db.commit()

    # 重写该学科的 chunk_meta.json（不含向量，快）
    if chunk.subject_id:
        try:
            from app.services.rag_service import RAGService
            await RAGService.__new__(RAGService)._write_chunk_meta(db, chunk.subject_id)
        except Exception as e:
            logger.warning("更新分块章节后重写索引元数据失败: %s", e)

    return {"message": "已更新分块章节", "chunk_id": chunk_id, "chapter_id": chapter_id, "section_id": section_id}


@router.post("/chunks/rebuild-index")
async def rebuild_kb_index(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """Rebuild knowledge base FAISS index for a subject (问答检索用的层级 chunk 索引)."""
    from app.services.rag_service import RAGService
    from app.services.gpu_manager import gpu_manager

    gpu_manager.to_gpu("embedding")  # 构建索引需要 embedding 在 GPU；clear_gpu 会把它挪到 CPU 导致跑满 CPU
    try:
        rag = RAGService()
        result = await rag.build_chunk_index(subject_id, db=db)
        return {"message": "知识库索引已重建", "result": result}
    finally:
        gpu_manager.restore_defaults()


@router.post("/rebuild-index")
async def rebuild_search_index(
    subject_id: int = Query(..., description="Subject ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_teacher_or_admin),
):
    """重建指定学科用于问答检索的知识库分块索引。"""
    from app.services.rag_service import RAGService
    from app.services.gpu_manager import gpu_manager

    gpu_manager.to_gpu("embedding")  # 构建索引需要 embedding 在 GPU；clear_gpu 会把它挪到 CPU 导致跑满 CPU
    try:
        rag = RAGService()
        kb = await rag.build_chunk_index(subject_id, db=db)
        return {"message": "学科知识库索引已重建", "kb": kb}
    finally:
        gpu_manager.restore_defaults()


# ── Build tree from parsed document ──

@router.post("/build-from-doc/{doc_id}")
async def build_tree_from_document(
    doc_id: int,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """从解析文档构建知识库内容块（知识树已改为手动构建，不再由文档自动生成）。"""
    # Find the subject this document is assigned to
    from sqlalchemy import text as sa_text
    subj_result = await db.execute(
        sa_text("SELECT subject_id FROM document_subjects WHERE document_id = :did LIMIT 1"),
        {"did": doc_id}
    )
    subj_row = subj_result.fetchone()
    subject_id = subj_row[0] if subj_row else None

    if not subject_id:
        raise HTTPException(400, "请先将文档分配到学科后再构建知识库")

    # Build content chunks (knowledge base) from content_list_v2.json
    result = await tree_service.build_from_content_list(doc_id, subject_id, db, is_primary=True)

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

    structure = tree_service.preview_structure(doc_id)
    return structure


# ── Stats ──

@router.get("/stats")
async def get_tree_stats(db: AsyncSession = Depends(get_db)):
    """Get knowledge tree statistics."""
    total_kps = await db.scalar(select(func.count(KnowledgePoint.id)))
    total_chunks = await db.scalar(select(func.count(ContentChunk.id)))
    total_ex = await db.scalar(select(func.count(TestQuestion.id)))
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
    subject_id: int = Query(..., description="Subject ID"),
    chapter: str | None = None,
    has_summary: bool | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_admin_user),
):
    """Get all KPs with summary, chunk count, and exercise count for review."""
    result = await db.execute(
        select(KnowledgePoint)
        .where(KnowledgePoint.subject_id == subject_id)
        .order_by(KnowledgePoint.sort_order)
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
            select(TestQuestion.kp_id, sa_func.count(TestQuestion.id))
            .where(TestQuestion.kp_id.in_(kp_ids))
            .group_by(TestQuestion.kp_id)
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
            select(TestQuestion).where(TestQuestion.kp_id.like(f"{sec_id}.%"))
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
