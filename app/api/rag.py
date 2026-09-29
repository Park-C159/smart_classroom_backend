"""Chat API — SSE streaming RAG-based Q&A + speech transcription."""
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
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
    subject_id: Optional[int] = Form(None),
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
            # 跑在线程里，避免同步 LLM 调用（含限流重试的 sleep）阻塞事件循环
            resolved = await asyncio.to_thread(llm.resolve_history, final_question, chat_history)
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

        # 解析学科：显式传入 > 用户已分配学科（取第一个）> 从所选教材推导
        user_id = current_user.get("user_id") or current_user.get("id")
        resolved_subject_id = subject_id
        if not resolved_subject_id:
            from app.models import UserSubject
            sids = (await db.execute(
                select(UserSubject.subject_id).where(UserSubject.user_id == user_id)
            )).scalars().all()
            if sids:
                resolved_subject_id = sids[0]
                logger.info("使用用户已分配学科: %d", resolved_subject_id)

        # 学生未分配学科时禁止使用答疑（学科由管理员/教师在用户管理中分配）
        if not resolved_subject_id and current_user.get("role") == "student":
            def err_gen():
                yield json.dumps({"type": "error", "content": "您尚未分配学科，暂无法使用答疑，请联系管理员分配学科"}, ensure_ascii=False) + "\n\n"
            return StreamingResponse(err_gen(), media_type="text/event-stream")

        # 自动选教材：优先该学科下的全部教材
        if not doc_ids_to_search:
            cond = select(DocumentSubject.document_id)
            if resolved_subject_id:
                cond = cond.where(DocumentSubject.subject_id == resolved_subject_id)
            all_ids = [r[0] for r in (await db.execute(cond)).all()]
            if all_ids:
                doc_ids_to_search = all_ids
                logger.info("Auto-selected documents (subject=%s): %s", resolved_subject_id, all_ids)

        # 若仍无学科，从所选教材推导
        if not resolved_subject_id and doc_ids_to_search:
            subj_rows = await db.execute(
                select(DocumentSubject.subject_id)
                .where(DocumentSubject.document_id.in_(doc_ids_to_search))
                .distinct()
            )
            distinct_sids = [r[0] for r in subj_rows.all() if r[0] is not None]
            if len(distinct_sids) == 1:
                resolved_subject_id = distinct_sids[0]
            elif len(distinct_sids) > 1:
                def err_gen():
                    yield json.dumps({"type": "error", "content": "所选教材跨越多个学科，请只选择同一学科的教材"}, ensure_ascii=False) + "\n\n"
                return StreamingResponse(err_gen(), media_type="text/event-stream")

        if not doc_ids_to_search:
            def err_gen():
                yield json.dumps({"type": "error", "content": "未选择任何教材，请先上传教材并分配学科"}, ensure_ascii=False) + "\n\n"
            return StreamingResponse(err_gen(), media_type="text/event-stream")

        result = await db.execute(select(Document).where(Document.id.in_(doc_ids_to_search)))
        docs = result.scalars().all()
        doc_title_map = {d.id: d.filename for d in docs}

        # 学科专属 system prompt（按学科定制 + 强调优先参考 RAG 召回内容）
        subject_name = None
        if resolved_subject_id:
            from app.models import Subject
            subject_name = await db.scalar(
                select(Subject.name).where(Subject.id == resolved_subject_id)
            )
        system_prompt = LLMService.build_subject_prompt(subject_name)

        all_candidates = []
        final_top_k = top_k
        if hierarchical and resolved_subject_id:
            if not rag.has_chunk_index(resolved_subject_id):
                def err_gen():
                    yield json.dumps({"type": "error", "content": "该学科尚未构建知识库，暂不支持答疑"}, ensure_ascii=False) + "\n\n"
                return StreamingResponse(err_gen(), media_type="text/event-stream")
            candidates = rag.retrieve_chunks(
                final_question, subject_id=resolved_subject_id,
                top_k=top_k,  # 只检索知识库（扁平分块）
            )
            for c in candidates:
                c["doc_id"] = c.get("source_doc_id", 1)
                c["doc_title"] = doc_title_map.get(c["doc_id"], f"文档{c['doc_id']}")
            all_candidates = candidates

            final_top_k = top_k

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
                for kind, text in llm.get_stream_response(query=final_question, context=ctx, system_prompt=system_prompt, history=chat_history, deep_think=deep_think):
                    if text:
                        yield json.dumps({"type": kind, "content": text}, ensure_ascii=False) + "\n\n"
                yield json.dumps({"type": "sources", "sources": [], "web_sources": web_sources}, ensure_ascii=False) + "\n\n"
                yield json.dumps({"type": "done"}, ensure_ascii=False) + "\n\n"
            return StreamingResponse(no_ctx_gen(), media_type="text/event-stream")

        # Dedup KB（分块检索结果去重）
        kb_unique = {}
        for c in all_candidates:
            key = c.get("chunk_id")
            score = c.get("rerank_score", c.get("score", 0))
            if key not in kb_unique or score > kb_unique[key].get("rerank_score", 0):
                kb_unique[key] = c
        retrieved = sorted(kb_unique.values(), key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

        # 提取与问题相关的知识点（相关性 = 重排分数），用于学情判断；章 + 节两级
        kp_relevance = {}
        sec_relevance = {}
        for c in retrieved:
            score = c.get("rerank_score", c.get("score", 0)) or 0
            kp_id = c.get("kp_id")
            if kp_id and kp_id != "flat":
                kp_relevance[kp_id] = max(kp_relevance.get(kp_id, 0), float(score))
            section_id = c.get("section_id")
            if section_id:
                sec_relevance[section_id] = max(sec_relevance.get(section_id, 0), float(score))
        matched_kps = ({
            "kps": [{"kp_id": k, "relevance": round(v, 3)}
                    for k, v in sorted(kp_relevance.items(), key=lambda x: x[1], reverse=True)][:8],
            "sections": [{"kp_id": s, "relevance": round(v, 3)}
                         for s, v in sorted(sec_relevance.items(), key=lambda x: x[1], reverse=True)][:8],
            "count": len(kp_relevance),
        } if (kp_relevance or sec_relevance) else None)

        # Fetch FULL original chunk content from DB for sources
        kb_ids = list(set(
            [c.get("chunk_id") for c in retrieved if c.get("chunk_id")]
        ))
        kb_content_map = {}
        if kb_ids:
            from app.models import ContentChunk as CC
            db_chunks = await db.execute(select(CC).where(CC.id.in_(kb_ids)))
            for dc in db_chunks.scalars().all():
                kb_content_map[dc.id] = dc.content

        sources = []
        for i, chunk in enumerate(retrieved):
            # DB content first (source of truth), fallback to FAISS metadata
            full_text = (kb_content_map.get(chunk.get("chunk_id"))
                         or chunk.get("full_text")
                         or chunk.get("text", ""))
            excerpt = _safe_truncate(full_text, 200)
            src = {
                "id": i + 1,
                "excerpt": excerpt,
                "content_full": full_text,
                "kb_id": chunk.get("chunk_id"),
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

        # 记录互动（flush 拿到 interaction_id，随 done 事件返回前端，用于打分反馈→学情）
        user_id = current_user.get("user_id") or current_user.get("id")
        interaction_id = None
        try:
            log = InteractionLog(
                user_id=user_id,
                question=question[:500],
                matched_kps=matched_kps,
            )
            db.add(log)
            await db.flush()
            interaction_id = log.id
        except Exception as e:
            logger.warning("Failed to log interaction: %s", e)

        def stream_gen():
            try:
                # Send content first so user sees answer immediately
                # Pass original chat history (user/assistant pairs) for context
                llm_context = list(retrieved)
                if web_context_item:
                    llm_context = [web_context_item] + llm_context
                for kind, text in llm.get_stream_response(
                    query=final_question, context=llm_context,
                    system_prompt=system_prompt,
                    history=chat_history, deep_think=deep_think,
                ):
                    if text:
                        yield json.dumps({"type": kind, "content": text}, ensure_ascii=False) + "\n\n"
                # Sources at the end — avoid blocking content with large JSON
                yield json.dumps({"type": "sources", "sources": sources, "web_sources": web_sources}, ensure_ascii=False) + "\n\n"
                yield json.dumps({"type": "done", "interaction_id": interaction_id}, ensure_ascii=False) + "\n\n"
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
        result = await asyncio.to_thread(llm.get_sync_response, prompt, max_tokens=20)
        topic = result.strip().replace('"', '').replace('"', '').replace('"', '')
        return {"topic": topic[:20]}
    except Exception as e:
        logger.error("生成主题失败: %s", e)
        return {"topic": question[:20]}


class FeedbackRequest(BaseModel):
    feedback: str | None = None  # helpful | not_helpful | None(取消)


@router.post("/interactions/{interaction_id}/feedback")
async def rate_interaction(
    interaction_id: int,
    body: FeedbackRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """学生对某次答疑打「有帮助/没帮助」；再点一次相同反馈则取消（写 None）。"""
    log = (await db.execute(
        select(InteractionLog).where(InteractionLog.id == interaction_id)
    )).scalar_one_or_none()
    if not log:
        raise HTTPException(404, "互动记录不存在")
    if log.user_id != current_user["user_id"]:
        raise HTTPException(403, "无权操作该记录")
    if body.feedback not in (None, "helpful", "not_helpful"):
        raise HTTPException(400, "无效反馈")
    log.feedback = body.feedback
    await db.commit()
    return {
        "message": "反馈已记录" if body.feedback else "反馈已取消",
        "interaction_id": interaction_id,
        "feedback": body.feedback,
    }
