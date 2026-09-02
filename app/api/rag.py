"""Chat API — SSE streaming RAG-based Q&A + speech transcription."""
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.security import get_current_user
from app.models import Document, DocumentSubject, InteractionLog
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.file_processor import FileProcessor
from app.services.web_search import web_search_service

logger = logging.getLogger(__name__)


def _safe_truncate(text: str, max_len: int = 200) -> str:
    """Truncate text without cutting through LaTeX formulas."""
    if len(text) <= max_len:
        return text
    cut = text[:max_len]
    # Extend to close unclosed $$
    last_ss = cut.rfind("$$")
    if last_ss != -1:
        close_ss = text.find("$$", last_ss + 2)
        if close_ss == -1 or close_ss > last_ss:
            # Count $$ — if odd, extend to next $$
            count = cut.count("$$")
            if count % 2 != 0:
                nxt = text.find("$$", last_ss + 2)
                if nxt != -1 and nxt - max_len < 300:
                    cut = text[:nxt + 2]
    # Extend to close unclosed $
    sgl_count = cut.count("$") - cut.count("$$") * 2
    if sgl_count % 2 != 0:
        nxt = text.find("$", cut.rfind("$") + 1)
        if nxt != -1 and nxt - max_len < 100:
            cut = text[:nxt + 1]
    # Prefer sentence boundary
    for punct in ("。", "；", "\n", "，"):
        idx = cut.rfind(punct)
        if idx > max_len * 0.6:
            cut = cut[:idx + 1]
            break
    return cut + ("…" if len(cut) < len(text) else "")


# ── 图片识别 ──
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


def _is_image(file: UploadFile) -> bool:
    """Determine whether an uploaded file is an image."""
    if file.content_type and file.content_type.startswith("image/"):
        return True
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    return suffix in IMAGE_EXTENSIONS


# 限制图片识别并发数，避免多张图同时打爆 mineru-api / 显存
_image_recognize_semaphore = asyncio.Semaphore(2)


async def _recognize_image(file: UploadFile) -> str:
    """识别上传图片 → 文字 + LaTeX 公式 + 图形描述。

    优先用本地 MinerU VLM（vlm-engine，含图形理解），失败回退 pipeline OCR。
    受 _image_recognize_semaphore 限制：最多 2 张图同时识别，超出排队等待。
    """
    async with _image_recognize_semaphore:
        try:
            # ① 本地 VLM：文字 + 公式 + 图形描述
            text = await _file_processor.extract_image_text(file)
            if text:
                return text
            logger.info("图片 VLM 识别结果为空，回退 pipeline OCR")
        except Exception as e:
            logger.warning("图片 VLM 识别失败，回退 pipeline OCR: %s", e)
        # ② 回退：pipeline OCR（文字 + 公式，无图形）
        try:
            await file.seek(0)
            return await _file_processor.extract_text(file, enable_formula=True, enable_table=False)
        except Exception as e:
            logger.error("图片 MinerU 识别失败: %s", e)
            return ""


router = APIRouter(prefix="/api/chat", tags=["问答"])

_rag_service: Optional[RAGService] = None
_llm_service: Optional[LLMService] = None
_file_processor = FileProcessor()


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


@router.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Convert uploaded audio to text via Whisper API."""
    try:
        from app.services.stt_service import STTService
        stt = STTService()
        text = await stt.transcribe(file)
        return {"text": text}
    except ValueError as e:
        raise HTTPException(400, f"语音服务未配置: {e}")
    except Exception as e:
        logger.error("语音识别失败: %s", e)
        raise HTTPException(500, f"语音识别失败: {str(e)}")


@router.post("/recognize-image")
async def recognize_image_endpoint(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """识别上传的图片 → 文字 + LaTeX 公式（上传时解析，供前端合并进提问）。"""
    if not _is_image(file):
        raise HTTPException(400, "仅支持图片文件")
    text = await _recognize_image(file)
    if not text:
        raise HTTPException(400, "图片识别失败，请换一张更清晰的图片")
    return {"text": text}


@router.post("/stream")
async def chat_stream(
    question: Optional[str] = Form(None),
    doc_id: int = Form(0),
    top_k: Optional[int] = Form(5),
    selected_doc_ids: Optional[str] = Form(None),
    hierarchical: bool = Form(True),
    history: Optional[str] = Form(None),
    deep_think: bool = Form(False),
    smart_search: bool = Form(False),
    files: List[UploadFile] = File(None),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Main RAG Q&A endpoint with SSE streaming."""
    if not question and not files:
        raise HTTPException(400, "请提供问题或上传文件")

    question = question or ""
    logger.info("📨 收到请求: question=%.30s, hierarchical=%s, deep_think=%s, smart_search=%s, files=%d",
                question, hierarchical, deep_think, smart_search, len(files) if files else 0)

    # 1. Parse document IDs
    doc_ids_to_search = []
    if selected_doc_ids:
        try:
            doc_ids_to_search = json.loads(selected_doc_ids)
        except Exception:
            doc_ids_to_search = []
    if not doc_ids_to_search and doc_id != 0:
        doc_ids_to_search = [doc_id]

    # 2. Process uploaded files（图片 → VLM/MinerU 识别成文字和公式；其他 → MinerU）
    # 注：MinerU 跑在独立 mineru-api 进程（或临时子进程），不占用后端 GPU，
    #     因此这里无需 clear_gpu/restore（否则会和并发的 RAG 检索抢模型状态）。
    extracted_texts = []
    if files:
        for file in files:
            if not file.filename:
                continue
            try:
                if _is_image(file):
                    text = await _recognize_image(file)
                    if not text:
                        raise Exception("图片识别失败")
                    label = f"【图片: {file.filename}】"
                else:
                    text = await _file_processor.extract_text(file)
                    label = f"【文件: {file.filename}】"
            except Exception as e:
                logger.error("文件 %s 解析失败: %s", file.filename, e)
                extracted_texts.append(f"【文件: {file.filename}】\n[解析失败: {e}]")
                continue
            if len(text) > 4000:
                text = text[:4000] + "\n...（内容过长已截断）"
            extracted_texts.append(f"{label}\n{text}")

    final_question = question
    if extracted_texts:
        final_question = f"{question}\n\n【用户上传文件内容】\n{'\n\n'.join(extracted_texts)}"

    # Parse conversation history
    chat_history = []
    if history:
        try:
            chat_history = json.loads(history)
        except Exception:
            pass

    # 4. RAG + streaming
    try:
        rag = get_rag_service()
        llm = get_llm_service()

        # History agent: check if history is needed and rewrite question
        history_context = ""
        if chat_history:
            resolved = llm.resolve_history(final_question, chat_history)
            if resolved["needed"]:
                history_context = resolved["context"]
                if resolved["rewritten_question"] != final_question:
                    logger.info("Question rewritten: %.50s → %.50s", final_question, resolved["rewritten_question"])
                    # Use rewritten question for retrieval, original for display
                    final_question = resolved["rewritten_question"]

        # 智能搜索：联网搜索结果作为额外上下文（供 LLM 回答参考）
        web_context_item = None
        web_results = []
        if smart_search:
            try:
                web_results = await asyncio.to_thread(web_search_service.search, final_question, 5)
            except Exception as e:
                logger.warning("智能搜索异常: %s", e)
                web_results = []
            if web_results:
                web_text = "\n\n".join(
                    f"【{i + 1}】{r['title']}\n{r['snippet']}\n{r['url']}"
                    for i, r in enumerate(web_results)
                )
                web_context_item = {"text": f"（以下为联网搜索结果，供回答参考）\n{web_text}", "source": "web"}
                logger.info("智能搜索：获取 %d 条联网结果", len(web_results))

        # 联网搜索来源（用于前端来源面板展示，可点击跳转）
        web_sources = [
            {"title": r["title"], "url": r["url"], "snippet": r["snippet"]}
            for r in web_results
        ]

        # Auto-select: if no docs specified but only one subject exists, use all its docs
        if not doc_ids_to_search:
            from app.models import DocumentSubject
            # Find all docs with any assigned subject
            all_docs = await db.execute(select(DocumentSubject.document_id))
            all_ids = [r[0] for r in all_docs.all()]
            if all_ids:
                doc_ids_to_search = all_ids
                logger.info("Auto-selected documents: %s", all_ids)

        if not doc_ids_to_search:
            def err_gen():
                yield json.dumps({"type": "error", "content": "未选择任何教材，请先上传教材并分配学科"}, ensure_ascii=False) + "\n\n"
            return StreamingResponse(err_gen(), media_type="text/event-stream")

        result = await db.execute(select(Document).where(Document.id.in_(doc_ids_to_search)))
        docs = result.scalars().all()
        doc_title_map = {d.id: d.filename for d in docs}

        all_candidates = []
        final_top_k = top_k
        if hierarchical:
            candidates = rag.retrieve_hierarchical(
                final_question, subject_id=9,
                top_k=3,  # KB: 3 chunks
            )
            for c in candidates:
                c["doc_id"] = c.get("source_doc_id", 1)
                c["doc_title"] = doc_title_map.get(c["doc_id"], f"文档{c['doc_id']}")
            all_candidates = candidates

            # QB search — 5 results, merged with KB
            import sys as _sys
            qb_results = rag.search_qb(final_question, subject_id=9, top_k=3)
            _sys.stderr.write(f"DEBUG QB search: {len(qb_results)} results\n")
            _sys.stderr.flush()
            for qb in qb_results:
                qb["doc_id"] = qb.get("source_doc_id", 2)
                qb["doc_title"] = doc_title_map.get(qb["doc_id"], f"文档{qb['doc_id']}")
                qb["chapter_title"] = qb.get("chapter", "")
                qb["source"] = "qb"
                qb["chunk_type"] = "exercise"
                qb["text"] = qb.get("question_text", qb.get("text", ""))[:512]
                qb["page_number"] = qb.get("page_num")
                all_candidates.append(qb)
            logger.info("QB search added %d results", len(qb_results))

            final_top_k = 6

        if not all_candidates:
            # Fallback to two-stage or single-index retrieval
            for did in doc_ids_to_search:
                subj_result = await db.execute(
                    select(DocumentSubject.subject_id).where(DocumentSubject.document_id == did)
                )
                subj_id = subj_result.scalar_one_or_none()
                candidates = rag.retrieve_two_stage(final_question, top_k=top_k * 4, doc_id=did, subject_id=subj_id)
                if not candidates:
                    if rag.doc_id != did or rag.index is None:
                        if not rag.load_index(did):
                            logger.warning("⚠️ 跳过文档 %d（索引加载失败）", did)
                            continue
                    candidates = rag.retrieve(final_question, top_k * 2)
                for c in candidates:
                    c["doc_id"] = did
                    c["doc_title"] = doc_title_map.get(did, f"文档{did}")
                all_candidates.extend(candidates)

        if not all_candidates:
            def no_ctx_gen():
                ctx = [web_context_item] if web_context_item else []
                for kind, text in llm.get_stream_response(query=final_question, context=ctx, history=chat_history, deep_think=deep_think):
                    if text:
                        yield json.dumps({"type": kind, "content": text}, ensure_ascii=False) + "\n\n"
                yield json.dumps({"type": "sources", "sources": [], "web_sources": web_sources}, ensure_ascii=False) + "\n\n"
                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
            return StreamingResponse(no_ctx_gen(), media_type="text/event-stream")

        # Split KB and QB, each already reranked internally
        kb_candidates = [c for c in all_candidates if c.get("source") != "qb"]
        qb_candidates = [c for c in all_candidates if c.get("source") == "qb"]

        # Dedup KB
        kb_unique = {}
        for c in kb_candidates:
            key = (c.get("doc_id"), c.get("kp_id", ""), c.get("chunk_type", ""))
            score = c.get("rerank_score", c.get("score", 0))
            if key not in kb_unique or score > kb_unique[key].get("rerank_score", 0):
                kb_unique[key] = c
        kb_list = sorted(kb_unique.values(), key=lambda x: x.get("rerank_score", 0), reverse=True)[:3]

        # Dedup QB
        qb_unique = {}
        for c in qb_candidates:
            key = (c.get("doc_id"), c.get("id", ""), c.get("chunk_type", ""))
            score = c.get("rerank_score", c.get("score", 0))
            if key not in qb_unique or score > qb_unique[key].get("rerank_score", 0):
                qb_unique[key] = c
        qb_list = sorted(qb_unique.values(), key=lambda x: x.get("rerank_score", 0), reverse=True)[:3]

        retrieved = kb_list + qb_list

        # Fetch FULL original chunk content from DB for sources
        # Separate KB (ContentChunk) and QB (QuestionBank) — their IDs overlap!
        kb_ids = list(set(
            [c.get("chunk_id") for c in retrieved if c.get("chunk_id")]
        ))
        qb_ids = list(set(
            [c.get("id") for c in retrieved if c.get("source") == "qb" and c.get("id")]
        ))
        kb_content_map = {}
        qb_content_map = {}
        if kb_ids:
            from app.models import ContentChunk as CC
            db_chunks = await db.execute(select(CC).where(CC.id.in_(kb_ids)))
            for dc in db_chunks.scalars().all():
                kb_content_map[dc.id] = dc.content
        if qb_ids:
            from app.models import QuestionBank as QB
            qb_entries = await db.execute(select(QB).where(QB.id.in_(qb_ids)))
            for qb in qb_entries.scalars().all():
                qb_content_map[qb.id] = qb.question_text or ""

        sources = []
        for i, chunk in enumerate(retrieved):
            if chunk.get("source") == "qb":
                full_text = qb_content_map.get(chunk.get("id")) or chunk.get("text", "")
            else:
                # DB content first (source of truth), fallback to FAISS metadata
                full_text = (kb_content_map.get(chunk.get("chunk_id"))
                             or chunk.get("full_text")
                             or chunk.get("text", ""))
            excerpt = _safe_truncate(full_text, 200)
            src = {
                "id": i + 1,
                "excerpt": excerpt,
                "content_full": full_text,
                "kb_id": chunk.get("chunk_id"),  # for KB chunks, None for QB
                "qb_id": chunk.get("id") if chunk.get("source") == "qb" else None,
                "chunk_type": chunk.get("chunk_type", ""),
                "source": chunk.get("source", "page"),
                "doc_title": chunk.get("doc_title", ""),
                "doc_id": chunk.get("doc_id", chunk.get("source_doc_id", 1)),
                "chapter": chunk.get("chapter_title", ""),
                "section": chunk.get("section_title", ""),
            }
            if "page_number" in chunk:
                src["page"] = chunk["page_number"]
            elif "page_num" in chunk:
                src["page"] = chunk["page_num"]
            if "adjacent_prev" in chunk:
                src["adjacent_prev"] = chunk["adjacent_prev"]
            if "adjacent_next" in chunk:
                src["adjacent_next"] = chunk["adjacent_next"]
            # QB exercise answers
            if chunk.get("source") == "qb" and chunk.get("answer_text"):
                src["answer_text"] = chunk["answer_text"]
            sources.append(src)

        # Log interaction (fire-and-forget, don't block response)
        user_id = current_user.get("user_id") or current_user.get("id")

        # Schedule interaction logging — runs after streaming without blocking
        asyncio.ensure_future(_log_interaction(user_id, question, sources))

        def stream_gen():
            try:
                # Send content first so user sees answer immediately
                # Pass original chat history (user/assistant pairs) for context
                llm_context = list(retrieved)
                if web_context_item:
                    llm_context = [web_context_item] + llm_context
                for kind, text in llm.get_stream_response(
                    query=final_question, context=llm_context,
                    history=chat_history, deep_think=deep_think,
                ):
                    if text:
                        yield json.dumps({"type": kind, "content": text}, ensure_ascii=False) + "\n\n"
                # Sources at the end — avoid blocking content with large JSON
                yield json.dumps({"type": "sources", "sources": sources, "web_sources": web_sources}, ensure_ascii=False) + "\n\n"
                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
            except Exception as e:
                yield json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False) + "\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    except Exception as e:
        logger.error("❌ 错误: %s", e, exc_info=True)
        def err_gen():
            yield json.dumps({"type": "error", "content": f"服务错误: {str(e)[:300]}"}, ensure_ascii=False) + "\n\n"
            yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
        return StreamingResponse(err_gen(), media_type="text/event-stream")


@router.post("/generate-topic")
async def generate_topic(
    question: str = Form(""),
    answer: str = Form(""),
):
    """Generate a short topic title for a conversation."""
    llm = get_llm_service()
    prompt = (
        f"用户问题：{question[:200]}\n"
        f"助手回答（摘要）：{answer[:300]}\n\n"
        "请用5-10个汉字概括这段对话的主题，只输出主题，不要标点符号和其他内容。"
    )
    try:
        result = llm.get_sync_response(prompt, max_tokens=20)
        topic = result.strip().replace('"', '').replace('"', '').replace('"', '')
        return {"topic": topic[:20]}
    except Exception as e:
        logger.error("生成主题失败: %s", e)
        return {"topic": question[:20]}


async def _log_interaction(user_id: int, question: str, sources: list):
    """Log a Q&A interaction asynchronously (own DB session)."""
    try:
        from app.core.database import async_session_factory
        async with async_session_factory() as db2:
            chapter_ids = list(set(
                s.get("chapter", "") for s in sources if s.get("chapter")
            ))
            log = InteractionLog(
                user_id=user_id,
                question=question[:500],
                matched_kps={"chapters": chapter_ids, "count": len(sources)},
            )
            db2.add(log)
            await db2.commit()
            logger.info("Interaction logged: user=%d", user_id)
    except Exception as e:
        logger.warning("Failed to log interaction: %s", e)
