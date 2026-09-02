"""RAG Service — BGE-M3 embedding + FAISS index + BGE-Reranker."""
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Force offline mode — all models must be pre-downloaded
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import faiss
import numpy as np
import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class RAGService:
    """Singleton RAG pipeline: embedding → FAISS retrieval → reranker.

    Each document gets its own FAISS index stored under data/parsed/{doc_id}/.
    """

    _embedding_model: Optional[SentenceTransformer] = None
    _reranker_model: Optional[CrossEncoder] = None
    _initialized: bool = False

    def __init__(self):
        if not RAGService._initialized:
            self._init_models()

        self.embedding_model = RAGService._embedding_model
        self.reranker = RAGService._reranker_model
        self.index: Optional[faiss.Index] = None
        self.chunks: List[Dict] = []
        self.doc_id: Optional[int] = None

    @classmethod
    def _init_models(cls):
        if cls._initialized:
            return
        logger.info("🚀 加载 Embedding 模型: %s (fp16)", settings.BGE_M3_MODEL_NAME)
        cls._embedding_model = SentenceTransformer(
            settings.BGE_M3_MODEL_NAME,
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
        logger.info("🚀 加载 Reranker 模型: %s (fp16)", settings.RERANKER_MODEL_NAME)
        cls._reranker_model = CrossEncoder(
            settings.RERANKER_MODEL_NAME,
            local_files_only=True,
            model_kwargs={"torch_dtype": torch.float16},
        )
        cls._initialized = True
        logger.info("✅ RAG 模型加载完成 (fp16)")

    def _rerank_scores(self, query: str, texts: List[str]) -> List[float]:
        """计算 reranker 相关性分数（sigmoid(logit)，0-1）。

        sentence-transformers 5.6.1 的 CrossEncoder.predict 对 BGE-Reranker-v2-m3
        返回恒定值（bug），这里改用手动前向 + sigmoid，得到有区分度的分数。
        """
        pairs = [[query, t] for t in texts]
        device = next(self.reranker.model.parameters()).device
        feats = self.reranker.tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        feats = {k: v.to(device) for k, v in feats.items()}
        with torch.no_grad():
            logits = self.reranker.model(**feats).logits.flatten()
        return torch.sigmoid(logits).cpu().tolist()

    # ── helpers ──

    def _parsed_dir(self, doc_id: int) -> Path:
        return Path(settings.DATA_DIR) / "parsed" / str(doc_id)

    def _load_parsed_data(self, doc_id: int) -> Dict[str, Any]:
        result_file = self._parsed_dir(doc_id) / "result.json"
        if not result_file.exists():
            raise FileNotFoundError(f"文档 {doc_id} 的解析结果不存在: {result_file}")
        with open(result_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_metadata(self, text: str, page_num: int) -> Dict[str, str]:
        chapter = ""
        section = ""
        m = re.search(r"(第[一二三四五六七八九十百]+章|[Cc]hapter\s+\d+)", text)
        if m:
            chapter = m.group(0)
        m = re.search(r"(\d+\.\d+)\s+([^\n]+)", text)
        if m:
            section = f"{m.group(1)} {m.group(2)}"
        return {"page": str(page_num), "chapter": chapter, "section": section}

    # ── index build ──

    def build_index(self, doc_id: int) -> Dict[str, Any]:
        """Build FAISS index from MinerU parsed result.json (page-level chunks)."""
        data = self._load_parsed_data(doc_id)
        raw_data = data.get("raw_data", [])

        self.chunks = []

        if raw_data:
            # Group by page_idx
            page_groups: Dict[int, List[Dict]] = {}
            for item in raw_data:
                if item.get("type") in ("text", "equation") and item.get("text"):
                    page_idx = item.get("page_idx", 0)
                    page_groups.setdefault(page_idx, []).append(item)

            for page_idx, items in sorted(page_groups.items()):
                items.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])  # sort by y
                page_text = ""
                for item in items:
                    text = item.get("text", "")
                    if item.get("type") == "equation":
                        text = f"$${text}$$"
                    page_text += text + "\n\n"
                if page_text.strip():
                    self.chunks.append({
                        "chunk_id": f"{doc_id}_p{page_idx + 1}",
                        "page_num": page_idx + 1,
                        "text": page_text.strip(),
                        "doc_id": doc_id,
                        "metadata": self._extract_metadata(page_text, page_idx + 1),
                    })
        else:
            # Fallback: pages array
            for page in data.get("pages", []):
                page_num = page.get("page_num", 1)
                regions = page.get("regions", [])
                page_text = ""
                for region in regions:
                    if region.get("region_type") == "text":
                        page_text += region.get("content", "") + "\n\n"
                    elif region.get("region_type") == "formula":
                        page_text += f"$${region.get('content', '')}$$\n\n"
                if page_text.strip():
                    self.chunks.append({
                        "chunk_id": f"{doc_id}_p{page_num}",
                        "page_num": page_num,
                        "text": page_text.strip(),
                        "doc_id": doc_id,
                        "metadata": self._extract_metadata(page_text, page_num),
                    })

        if not self.chunks:
            raise ValueError("未能从解析结果中提取任何内容")

        texts = [c["text"] for c in self.chunks]
        embeddings = self.embedding_model.encode(texts, normalize_embeddings=True)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))

        self.doc_id = doc_id

        index_path = self._parsed_dir(doc_id) / "faiss.index"
        faiss.write_index(self.index, str(index_path))

        logger.info("📊 索引构建完成: doc_id=%d chunks=%d dim=%d", doc_id, len(self.chunks), dimension)
        return {"doc_id": doc_id, "total_chunks": len(self.chunks), "index_path": str(index_path)}

    # ── index load ──

    def load_index(self, doc_id: int) -> bool:
        """Load FAISS index and rebuild chunk list from parsed data."""
        self.doc_id = doc_id
        index_path = self._parsed_dir(doc_id) / "faiss.index"
        if not index_path.exists():
            logger.warning("索引文件不存在: %s", index_path)
            return False

        self.index = faiss.read_index(str(index_path))
        data = self._load_parsed_data(doc_id)
        raw_data = data.get("raw_data", [])
        self.chunks = []

        if raw_data:
            page_groups: Dict[int, List[Dict]] = {}
            for item in raw_data:
                if item.get("type") in ("text", "equation") and item.get("text"):
                    page_idx = item.get("page_idx", 0)
                    page_groups.setdefault(page_idx, []).append(item)
            for page_idx, items in sorted(page_groups.items()):
                items.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])
                page_text = ""
                for item in items:
                    text = item.get("text", "")
                    if item.get("type") == "equation":
                        text = f"$${text}$$"
                    page_text += text + "\n\n"
                if page_text.strip():
                    self.chunks.append({
                        "chunk_id": f"{doc_id}_p{page_idx + 1}",
                        "page_num": page_idx + 1,
                        "text": page_text.strip(),
                        "doc_id": doc_id,
                        "metadata": self._extract_metadata(page_text, page_idx + 1),
                    })
        else:
            for page in data.get("pages", []):
                page_num = page.get("page_num", 1)
                page_text = ""
                for region in page.get("regions", []):
                    if region.get("region_type") == "text":
                        page_text += region.get("content", "") + "\n\n"
                    elif region.get("region_type") == "formula":
                        page_text += f"$${region.get('content', '')}$$\n\n"
                if page_text.strip():
                    self.chunks.append({
                        "chunk_id": f"{doc_id}_p{page_num}",
                        "page_num": page_num,
                        "text": page_text.strip(),
                        "doc_id": doc_id,
                        "metadata": self._extract_metadata(page_text, page_num),
                    })

        logger.info("📂 索引加载完成: doc_id=%d chunks=%d", doc_id, len(self.chunks))
        return True

    # ── retrieval ──

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Vector retrieval → top-N candidates → reranker → final top_k."""
        if self.index is None:
            raise ValueError("索引未加载")
        if len(self.chunks) == 0:
            return []

        query_embedding = self.embedding_model.encode([query], normalize_embeddings=True)
        retrieve_k = min(top_k * 2, len(self.chunks))
        scores, indices = self.index.search(query_embedding.astype("float32"), retrieve_k)

        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                candidates.append({**self.chunks[idx], "score": float(score)})

        if not candidates:
            return []

        candidates.sort(key=lambda x: x["score"], reverse=True)
        max_rerank = min(len(candidates), 5)  # max 5 for reranker
        candidates = candidates[:max_rerank]

        if len(candidates) > 1:
            rerank_scores = self._rerank_scores(query, [c["text"] for c in candidates])
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    # ── Knowledge-tree-aware indices (two-stage RAG) ──

    async def build_knowledge_indices(self, doc_id: int) -> Dict[str, Any]:
        """Build 3 FAISS indices from knowledge tree data for two-stage retrieval.

        Creates:
          - faiss_kp.index: KP title + summary vectors for coarse retrieval
          - faiss_content.index: content chunk vectors with kp_id metadata
          - faiss_exercise.index: exercise vectors with kp_id metadata
        """
        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models import KnowledgePoint, ContentChunk, Exercise

        kps, chunks, exercises_list = [], [], []
        async with async_session_factory() as db:
            result = await db.execute(select(KnowledgePoint))
            kps = result.scalars().all()
            result = await db.execute(select(ContentChunk))
            chunks = result.scalars().all()
            result = await db.execute(select(Exercise).where(Exercise.faiss_id.is_(None)))
            exercises_list = result.scalars().all()

        # Build KP index
        kp_entries, content_entries, exercise_entries = [], [], []

        # KP summaries for stage-1 coarse retrieval
        for kp in kps:
            text = kp.title
            if kp.summary:
                text += f": {kp.summary}"
            kp_entries.append({
                "kp_id": kp.id,
                "text": text,
                "doc_id": doc_id,
            })

        # Content chunks for stage-2 fine retrieval
        for i, chunk in enumerate(chunks):
            content_entries.append({
                "chunk_id": chunk.id,
                "kp_id": chunk.kp_id,
                "chunk_type": chunk.chunk_type,
                "text": chunk.content,
                "doc_id": doc_id,
                "page_num": chunk.page_number,
            })
            chunk.faiss_id = i

        # Exercises for stage-2 exercise retrieval
        for i, ex in enumerate(exercises_list):
            exercise_entries.append({
                "exercise_id": ex.id,
                "kp_id": ex.kp_id,
                "text": ex.question_text,
                "doc_id": doc_id,
                "page_num": ex.page_number,
            })
            ex.faiss_id = i

        data_dir = self._parsed_dir(doc_id)

        # Build and save KP index
        if kp_entries:
            kp_embs = self.embedding_model.encode(
                [e["text"] for e in kp_entries], normalize_embeddings=True
            )
            kp_dim = kp_embs.shape[1]
            kp_index = faiss.IndexFlatIP(kp_dim)
            kp_index.add(kp_embs.astype("float32"))
            faiss.write_index(kp_index, str(data_dir / "faiss_kp.index"))
            with open(data_dir / "faiss_kp_meta.json", "w", encoding="utf-8") as f:
                json.dump(kp_entries, f, ensure_ascii=False)

        # Build and save content index
        if content_entries:
            ct_embs = self.embedding_model.encode(
                [e["text"] for e in content_entries], normalize_embeddings=True
            )
            ct_dim = ct_embs.shape[1]
            ct_index = faiss.IndexFlatIP(ct_dim)
            ct_index.add(ct_embs.astype("float32"))
            faiss.write_index(ct_index, str(data_dir / "faiss_content.index"))
            with open(data_dir / "faiss_content_meta.json", "w", encoding="utf-8") as f:
                json.dump(content_entries, f, ensure_ascii=False)

        # Build and save exercise index
        if exercise_entries:
            ex_embs = self.embedding_model.encode(
                [e["text"] for e in exercise_entries], normalize_embeddings=True
            )
            ex_dim = ex_embs.shape[1]
            ex_index = faiss.IndexFlatIP(ex_dim)
            ex_index.add(ex_embs.astype("float32"))
            faiss.write_index(ex_index, str(data_dir / "faiss_exercise.index"))
            with open(data_dir / "faiss_exercise_meta.json", "w", encoding="utf-8") as f:
                json.dump(exercise_entries, f, ensure_ascii=False)

        logger.info(
            "Two-stage indices built: KPs=%d, content=%d, exercises=%d",
            len(kp_entries), len(content_entries), len(exercise_entries),
        )
        return {
            "doc_id": doc_id,
            "kp_chunks": len(kp_entries),
            "content_chunks": len(content_entries),
            "exercise_chunks": len(exercise_entries),
        }

    async def build_subject_indices(self, subject_id: int, db=None) -> Dict[str, Any]:
        """Build/re-build per-subject FAISS indices (3 index files).

        Stores indices in data/vector_store/{subject_id}/ instead of per-document.
        """
        import os as _os
        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models import KnowledgePoint, ContentChunk, Exercise

        should_close = db is None
        if db is None:
            db = async_session_factory()

        try:
            # Collect all KPs, content chunks, exercises for this subject
            # (all KPs are subject-scoped; content and exercises linked via kp_id)
            kps = (await db.execute(select(KnowledgePoint))).scalars().all()
            chunks = (await db.execute(select(ContentChunk))).scalars().all()
            exercises_list = (await db.execute(select(Exercise))).scalars().all()

            store_dir = Path(settings.DATA_DIR) / "vector_store" / str(subject_id)
            store_dir.mkdir(parents=True, exist_ok=True)

            # KP index
            kp_entries = []
            for kp in kps:
                text = f"{kp.title}. {kp.summary or ''}"[:512]
                kp_entries.append({"kp_id": kp.id, "text": text, "title": kp.title})
            if kp_entries:
                embs = self.embedding_model.encode(
                    [e["text"] for e in kp_entries], normalize_embeddings=True
                )
                idx = faiss.IndexFlatIP(embs.shape[1])
                idx.add(embs.astype("float32"))
                faiss.write_index(idx, str(store_dir / "faiss_kp.index"))
                with open(store_dir / "faiss_kp_meta.json", "w", encoding="utf-8") as f:
                    json.dump(kp_entries, f, ensure_ascii=False)

            # Content index
            content_entries = []
            for i, c in enumerate(chunks):
                text = c.content[:512]
                content_entries.append({
                    "id": c.id, "kp_id": c.kp_id, "text": text,
                    "chunk_type": c.chunk_type, "faiss_id": i,
                    "page_num": c.page_number,  # PDF page number for source reference
                })
            if content_entries:
                embs = self.embedding_model.encode(
                    [e["text"] for e in content_entries], normalize_embeddings=True
                )
                idx = faiss.IndexFlatIP(embs.shape[1])
                idx.add(embs.astype("float32"))
                faiss.write_index(idx, str(store_dir / "faiss_content.index"))
                with open(store_dir / "faiss_content_meta.json", "w", encoding="utf-8") as f:
                    json.dump(content_entries, f, ensure_ascii=False)

            # Exercise index
            ex_entries = []
            for i, ex in enumerate(exercises_list):
                text = (ex.embedding_text or ex.question_text)[:512]
                ex_entries.append({
                    "id": ex.id, "kp_id": ex.kp_id, "text": text,
                    "question_type": ex.question_type, "faiss_id": i,
                    "page_num": ex.page_number,
                })
            if ex_entries:
                embs = self.embedding_model.encode(
                    [e["text"] for e in ex_entries], normalize_embeddings=True
                )
                idx = faiss.IndexFlatIP(embs.shape[1])
                idx.add(embs.astype("float32"))
                faiss.write_index(idx, str(store_dir / "faiss_exercise.index"))
                with open(store_dir / "faiss_exercise_meta.json", "w", encoding="utf-8") as f:
                    json.dump(ex_entries, f, ensure_ascii=False)

            logger.info(
                "Subject indices built (subject=%d): KPs=%d, content=%d, exercises=%d",
                subject_id, len(kp_entries), len(content_entries), len(ex_entries),
            )
            return {
                "subject_id": subject_id,
                "kp_chunks": len(kp_entries),
                "content_chunks": len(content_entries),
                "exercise_chunks": len(ex_entries),
            }
        finally:
            if should_close and db:
                await db.close()

    # ── Dual KB/QB index building ──

    async def build_kb_index(self, subject_id: int, db=None) -> Dict[str, Any]:
        """Build knowledge base FAISS index: content_chunks grouped by section."""
        import os as _os
        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models import ContentChunk, KnowledgePoint

        should_close = db is None
        if db is None:
            db = async_session_factory()
        try:
            store_dir = Path(settings.DATA_DIR) / "vector_store" / str(subject_id)
            store_dir.mkdir(parents=True, exist_ok=True)

            # Get all KB chunks with subject_id
            chunks = (await db.execute(
                select(ContentChunk).where(ContentChunk.subject_id == subject_id)
            )).scalars().all()

            kb_entries = []
            for i, c in enumerate(chunks):
                text = c.content[:512] if c.content else ""
                kb_entries.append({
                    "id": c.id, "kp_id": c.kp_id, "text": text,
                    "chunk_type": c.chunk_type, "faiss_id": i,
                    "page_num": c.page_number, "source_doc_id": c.source_doc_id,
                })

            if kb_entries:
                embs = self.embedding_model.encode(
                    [e["text"] for e in kb_entries], normalize_embeddings=True
                )
                idx = faiss.IndexFlatIP(embs.shape[1])
                idx.add(embs.astype("float32"))
                faiss.write_index(idx, str(store_dir / "faiss_kb.index"))
                with open(store_dir / "faiss_kb_meta.json", "w", encoding="utf-8") as f:
                    json.dump(kb_entries, f, ensure_ascii=False)

            logger.info("KB index: %d chunks (subject=%d)", len(kb_entries), subject_id)
            return {"kb_chunks": len(kb_entries)}
        finally:
            if should_close and db:
                await db.close()

    async def build_qb_index(self, subject_id: int, db=None) -> Dict[str, Any]:
        """Build question bank FAISS index: question_bank entries grouped by chapter."""
        import os as _os
        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models import QuestionBank

        should_close = db is None
        if db is None:
            db = async_session_factory()
        try:
            store_dir = Path(settings.DATA_DIR) / "vector_store" / str(subject_id)
            store_dir.mkdir(parents=True, exist_ok=True)

            qs = (await db.execute(
                select(QuestionBank).where(QuestionBank.subject_id == subject_id)
            )).scalars().all()

            qb_entries = []
            for i, q in enumerate(qs):
                text = (q.embedding_text or q.question_text)[:512]
                qb_entries.append({
                    "id": q.id, "chapter": q.chapter, "text": text,
                    "question_type": q.question_type, "faiss_id": i,
                    "page_num": q.page_number, "source_doc_id": q.source_doc_id,
                    "source": q.source,
                    "question_text": (q.question_text or "")[:1000],
                    "answer_text": (q.answer_text or "")[:2000],
                })

            if qb_entries:
                embs = self.embedding_model.encode(
                    [e["text"] for e in qb_entries], normalize_embeddings=True
                )
                idx = faiss.IndexFlatIP(embs.shape[1])
                idx.add(embs.astype("float32"))
                faiss.write_index(idx, str(store_dir / "faiss_qb.index"))
                with open(store_dir / "faiss_qb_meta.json", "w", encoding="utf-8") as f:
                    json.dump(qb_entries, f, ensure_ascii=False)

            logger.info("QB index: %d questions (subject=%d)", len(qb_entries), subject_id)
            return {"qb_questions": len(qb_entries)}
        finally:
            if should_close and db:
                await db.close()

    def search_qb(
        self, query: str, subject_id: int, top_k: int = 3, chapter_filter: str = None,
    ) -> List[Dict[str, Any]]:
        """Search question bank FAISS index for similar exercises.

        Returns entries with question_text, answer_text for LLM reference.
        """
        store_dir = self._subject_index_dir(subject_id)
        idx_path = store_dir / "faiss_qb.index"
        meta_path = store_dir / "faiss_qb_meta.json"

        if not idx_path.exists() or not meta_path.exists():
            logger.warning("QB index not found at %s", store_dir)
            return []

        with open(meta_path, "r", encoding="utf-8") as f:
            qb_meta = json.load(f)

        if not qb_meta:
            return []

        idx = faiss.read_index(str(idx_path))
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        search_k = min(top_k * 5, idx.ntotal)
        scores, indices = idx.search(query_emb.astype("float32"), search_k)

        # 收集全部候选再重排（不提前截断到 top_k，提升召回）
        results = []
        for score, fi in zip(scores[0], indices[0]):
            if 0 <= fi < len(qb_meta):
                meta = dict(qb_meta[fi])
                if chapter_filter and meta.get("chapter") != chapter_filter:
                    continue
                meta["score"] = float(score)
                results.append(meta)

        # Rerank if we have enough candidates
        if len(results) > 1:
            try:
                rerank_scores = self._rerank_scores(query, [r["text"] for r in results])
                for i, s in enumerate(rerank_scores):
                    results[i]["rerank_score"] = float(s)
                results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            except Exception as e:
                logger.warning("QB rerank failed: %s", e)

        return results[:top_k]

    def _subject_index_dir(self, subject_id: int) -> Path:
        return Path(settings.DATA_DIR) / "vector_store" / str(subject_id)

    def _load_subject_indices(self, subject_id: int) -> Dict[str, Any]:
        """Load per-subject two-stage indices."""
        data_dir = self._subject_index_dir(subject_id)
        return self._load_indices_from_dir(data_dir)

    def _load_knowledge_indices(self, doc_id: int) -> Dict[str, Any]:
        """Load pre-built two-stage indices (legacy per-document fallback)."""
        data_dir = self._parsed_dir(doc_id)
        return self._load_indices_from_dir(data_dir)

    def _load_indices_from_dir(self, data_dir: Path) -> Dict[str, Any]:
        """Load index files from a directory."""
        result = {"kp": None, "content": None, "exercise": None}
        result = {"kp": None, "content": None, "exercise": None}

        kp_path = data_dir / "faiss_kp.index"
        if kp_path.exists():
            result["kp"] = faiss.read_index(str(kp_path))
            with open(data_dir / "faiss_kp_meta.json", "r", encoding="utf-8") as f:
                result["kp_meta"] = json.load(f)

        ct_path = data_dir / "faiss_content.index"
        if ct_path.exists():
            result["content"] = faiss.read_index(str(ct_path))
            with open(data_dir / "faiss_content_meta.json", "r", encoding="utf-8") as f:
                result["content_meta"] = json.load(f)

        ex_path = data_dir / "faiss_exercise.index"
        if ex_path.exists():
            result["exercise"] = faiss.read_index(str(ex_path))
            with open(data_dir / "faiss_exercise_meta.json", "r", encoding="utf-8") as f:
                result["exercise_meta"] = json.load(f)

        return result

    def retrieve_two_stage(
        self, query: str, top_k: int = 5, doc_id: Optional[int] = None,
        subject_id: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Two-stage retrieval: KP coarse → content+exercise fine → reranker.

        Stage 1: Query → faiss_kp.index → top-N matching KP IDs
        Stage 2: Query → faiss_content.index + faiss_exercise.index → filter by KP IDs
        Stage 3: Merge, deduplicate, reranker → final top_k
        """
        # Try subject-level indices first, fall back to doc-level
        indices = None
        if subject_id:
            indices = self._load_subject_indices(subject_id)
        if (not indices or not indices.get("kp")) and doc_id:
            indices = self._load_knowledge_indices(doc_id)
        if not indices or not indices.get("kp"):
            if doc_id is None:
                doc_id = self.doc_id
            if doc_id:
                indices = self._load_knowledge_indices(doc_id)
        if not indices or not indices.get("kp"):
            return self.retrieve(query, top_k) if doc_id else []

        if indices["kp"] is None:
            # Fall back to single-index retrieval
            if self.load_index(doc_id):
                return self.retrieve(query, top_k)
            return []

        kp_threshold = settings.RAG_KP_THRESHOLD
        max_candidates = settings.RAG_MAX_CANDIDATES

        # ── Stage 1: KP coarse retrieval ──
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)
        kp_top_k = min(settings.RAG_TOP_K_KP, len(indices["kp_meta"]))
        kp_scores, kp_indices = indices["kp"].search(
            query_emb.astype("float32"), kp_top_k
        )

        matched_kps = []
        for score, idx in zip(kp_scores[0], kp_indices[0]):
            if score >= kp_threshold and 0 <= idx < len(indices["kp_meta"]):
                matched_kps.append({
                    "kp_id": indices["kp_meta"][idx]["kp_id"],
                    "score": float(score),
                })
        logger.info("Stage 1: matched %d KPs (threshold=%.2f)", len(matched_kps), kp_threshold)
        matched_kp_ids = {m["kp_id"] for m in matched_kps}

        # ── Stage 2: Content + Exercise fine retrieval ──
        content_candidates = []
        if indices["content"] is not None and indices["content_meta"]:
            ct_pool = min(settings.RAG_CONTENT_CANDIDATES, len(indices["content_meta"]))
            ct_scores, ct_indices = indices["content"].search(
                query_emb.astype("float32"), ct_pool
            )
            for score, idx in zip(ct_scores[0], ct_indices[0]):
                if 0 <= idx < len(indices["content_meta"]):
                    meta = indices["content_meta"][idx]
                    if meta["kp_id"] in matched_kp_ids:
                        content_candidates.append({
                            **meta, "source": "content",
                            "score": float(score),
                        })

        exercise_candidates = []
        if indices["exercise"] is not None and indices["exercise_meta"]:
            ex_pool = min(settings.RAG_EXERCISE_CANDIDATES, len(indices["exercise_meta"]))
            ex_scores, ex_indices = indices["exercise"].search(
                query_emb.astype("float32"), ex_pool
            )
            for score, idx in zip(ex_scores[0], ex_indices[0]):
                if 0 <= idx < len(indices["exercise_meta"]):
                    meta = indices["exercise_meta"][idx]
                    if matched_kp_ids and meta["kp_id"] in matched_kp_ids:
                        exercise_candidates.append({
                            **meta, "source": "exercise",
                            "score": float(score),
                        })

        # Merge: content first, then exercises, deduplicate, cap at max_candidates
        candidates = content_candidates[:5] + exercise_candidates[:3]
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]
        if len(candidates) < top_k and not matched_kp_ids:
            # No KP match — add content results unfiltered
            for score, idx in zip(ct_scores[0], ct_indices[0]):
                if 0 <= idx < len(indices["content_meta"]):
                    meta = indices["content_meta"][idx]
                    candidates.append({**meta, "source": "content", "score": float(score)})
                if len(candidates) >= max_candidates:
                    break

        logger.info("Stage 2: %d candidates (content=%d, exercise=%d)",
                     len(candidates), len(content_candidates), len(exercise_candidates))

        if not candidates:
            return []

        # ── Stage 3: Reranker ──
        candidates.sort(key=lambda x: x["score"], reverse=True)
        if len(candidates) > 1 and len(candidates) <= 8:
            pairs = [[query, c["text"]] for c in candidates]
            rerank_scores = self.reranker.predict(pairs)
            for i, score in enumerate(rerank_scores):
                candidates[i]["rerank_score"] = float(score)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        return candidates[:top_k]

    # ── page content ──

    def get_page_content(self, doc_id: int, page_num: int) -> Dict[str, Any]:
        """Return parsed content for a single page."""
        data = self._load_parsed_data(doc_id)
        raw_data = data.get("raw_data", [])

        page_items = [item for item in raw_data if item.get("page_idx") == page_num - 1]
        page_items.sort(key=lambda x: x.get("bbox", [0, 0, 0, 0])[1])

        content = ""
        for item in page_items:
            if item.get("type") == "text":
                content += item.get("text", "") + "\n\n"
            elif item.get("type") == "equation":
                content += f"$${item.get('text', '')}$$\n\n"

        return {
            "page_num": page_num,
            "content": content.strip() or "（该页无可识别内容）",
            "total_pages": data.get("total_pages", 0),
            "doc_id": doc_id,
        }

    # ── Hierarchical RAG (chapter → section → chunk) ──

    async def build_hierarchical_index(self, subject_id: int, db=None) -> Dict[str, Any]:
        """Build chapter/section/chunk hierarchical FAISS indices.

        Creates:
          - faiss_chapter.index: chapter title vectors for coarse chapter matching
          - faiss_section.index: section title vectors for section matching within chapter
          - faiss_chunk.index: all chunk content vectors for fine retrieval
        """
        from sqlalchemy import select
        from app.core.database import async_session_factory
        from app.models import KnowledgePoint, ContentChunk

        should_close = db is None
        if db is None:
            db = async_session_factory()

        try:
            # Load all KPs and chunks
            kps = (await db.execute(
                select(KnowledgePoint).order_by(KnowledgePoint.sort_order)
            )).scalars().all()
            chunks = (await db.execute(
                select(ContentChunk)
            )).scalars().all()

            store_dir = Path(settings.DATA_DIR) / "vector_store" / str(subject_id)
            store_dir.mkdir(parents=True, exist_ok=True)

            # ── Chapter index ──
            chapter_entries = []
            section_entries = []

            for kp in kps:
                if kp.level == 0:
                    chapter_entries.append({
                        "kp_id": kp.id,
                        "title": kp.title,
                        "text": kp.title,
                        "chapter_num": self._extract_chapter_num(kp.id),
                        "sort_order": kp.sort_order,
                    })
                elif kp.level == 1:
                    # Find parent chapter
                    chapter_id = kp.parent_id
                    section_entries.append({
                        "kp_id": kp.id,
                        "title": kp.title,
                        "text": f"{kp.title}",
                        "chapter_id": chapter_id,
                        "sort_order": kp.sort_order,
                    })

            # Build chapter index
            if chapter_entries:
                ch_embs = self.embedding_model.encode(
                    [e["text"] for e in chapter_entries], normalize_embeddings=True
                )
                ch_idx = faiss.IndexFlatIP(ch_embs.shape[1])
                ch_idx.add(ch_embs.astype("float32"))
                faiss.write_index(ch_idx, str(store_dir / "faiss_chapter.index"))
                with open(store_dir / "chapter_meta.json", "w", encoding="utf-8") as f:
                    json.dump(chapter_entries, f, ensure_ascii=False)

            # Build section index
            if section_entries:
                sec_embs = self.embedding_model.encode(
                    [e["text"] for e in section_entries], normalize_embeddings=True
                )
                sec_idx = faiss.IndexFlatIP(sec_embs.shape[1])
                sec_idx.add(sec_embs.astype("float32"))
                faiss.write_index(sec_idx, str(store_dir / "faiss_section.index"))
                with open(store_dir / "section_meta.json", "w", encoding="utf-8") as f:
                    json.dump(section_entries, f, ensure_ascii=False)

            # ── Chunk index (all chunks for fine retrieval) ──
            # Build lookup: KP id → KP object
            kp_map = {k.id: k for k in kps}
            chunk_entries = []
            for i, c in enumerate(chunks):
                chapter = None
                section = None

                kp = kp_map.get(c.kp_id)
                if kp is not None:
                    # Exact KP match: use parent chain
                    if kp.level == 2:
                        parent = kp_map.get(kp.parent_id)
                        if parent:
                            section = parent
                            chapter = kp_map.get(parent.parent_id)
                    elif kp.level == 1:
                        section = kp
                        chapter = kp_map.get(kp.parent_id)
                else:
                    # kp_id like "KP-1.2.3": derive section "KP-1.2", chapter "KP-1"
                    import re as _re
                    m = _re.match(r'^(KP-\d+)\.(\d+)\.\d+$', c.kp_id or '')
                    if m:
                        section_id = f'{m.group(1)}.{m.group(2)}'
                        chapter_id = m.group(1)
                        section = kp_map.get(section_id)
                        chapter = kp_map.get(chapter_id)

                if chapter is None:
                    continue  # Can't place this chunk in the hierarchy

                entry = {
                    "chunk_id": c.id,
                    "kp_id": c.kp_id,
                    "chunk_type": c.chunk_type,
                    "text": c.content[:512],
                    "full_text": c.content,
                    "page_number": c.page_number,
                    "source_doc_id": c.source_doc_id,
                    "chapter_id": chapter.id,
                    "chapter_title": chapter.title,
                    "section_id": section.id if section else None,
                    "section_title": section.title if section else "",
                    "faiss_id": i,
                }
                chunk_entries.append(entry)

            if chunk_entries:
                ck_embs = self.embedding_model.encode(
                    [e["text"] for e in chunk_entries], normalize_embeddings=True
                )
                ck_idx = faiss.IndexFlatIP(ck_embs.shape[1])
                ck_idx.add(ck_embs.astype("float32"))
                faiss.write_index(ck_idx, str(store_dir / "faiss_chunk.index"))
                with open(store_dir / "chunk_meta.json", "w", encoding="utf-8") as f:
                    json.dump(chunk_entries, f, ensure_ascii=False)

            logger.info(
                "Hierarchical indices built: chapters=%d, sections=%d, chunks=%d",
                len(chapter_entries), len(section_entries), len(chunk_entries),
            )
            return {
                "subject_id": subject_id,
                "chapters": len(chapter_entries),
                "sections": len(section_entries),
                "chunks": len(chunk_entries),
            }
        finally:
            if should_close and db:
                await db.close()

    def retrieve_hierarchical(
        self, query: str, subject_id: int,
        top_k: int = 5, doc_filter: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Global RAG: KB + QB joint vector search → reranker. No chapter pre-filtering."""

        store_dir = self._subject_index_dir(subject_id)

        ck_idx_path = store_dir / "faiss_chunk.index"
        ck_meta_path = store_dir / "chunk_meta.json"

        if not ck_idx_path.exists():
            logger.warning("Chunk index not found at %s", ck_idx_path)
            return []

        with open(ck_meta_path, "r", encoding="utf-8") as f:
            chunk_meta = json.load(f)
        chunk_index = faiss.read_index(str(ck_idx_path))

        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)

        # ── KB: global vector search across all chunks ──
        search_k = min(settings.RAG_MAX_CANDIDATES * 3, chunk_index.ntotal)
        scores, indices = chunk_index.search(query_emb.astype("float32"), search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(chunk_meta):
                meta = dict(chunk_meta[idx])
                meta["score"] = float(score)
                meta["source"] = "kb"
                results.append(meta)
            if len(results) >= settings.RAG_MAX_CANDIDATES:
                break

        # ── QB: global search across all questions ──
        qb_idx_path = store_dir / "faiss_qb.index"
        qb_meta_path = store_dir / "faiss_qb_meta.json"
        if qb_idx_path.exists() and qb_meta_path.exists():
            with open(qb_meta_path, "r", encoding="utf-8") as f:
                qb_meta = json.load(f)
            qb_index = faiss.read_index(str(qb_idx_path))
            qb_search_k = min(settings.RAG_MAX_CANDIDATES * 2, qb_index.ntotal)
            qb_scores, qb_indices = qb_index.search(query_emb.astype("float32"), qb_search_k)
            for score, idx in zip(qb_scores[0], qb_indices[0]):
                if 0 <= idx < len(qb_meta):
                    meta = dict(qb_meta[idx])
                    meta["score"] = float(score)
                    meta["source"] = "qb"
                    meta["text"] = meta.get("question_text", meta.get("text", ""))[:512]
                    meta["chunk_type"] = "exercise"
                    meta["chapter_title"] = meta.get("chapter", "")
                    meta["page_number"] = meta.get("page_num")
                    results.append(meta)
                if sum(1 for r in results if r.get("source") == "qb") >= 3:
                    break
            logger.info("QB search: %d results", sum(1 for r in results if r.get("source") == "qb"))

        if not results:
            return []

        # ── Joint rerank (KB + QB together) ──
        TYPE_BOOST = {"theorem": 0.15, "definition": 0.15, "example": 0.1,
                       "proof": 0.1, "exercise": 0.1, "text": 0.05, "remark": 0.05}
        if len(results) > 1:
            try:
                rerank_scores = self._rerank_scores(query, [r["text"] for r in results[:12]])
                for i, s in enumerate(rerank_scores):
                    boost = TYPE_BOOST.get(results[i].get("chunk_type", ""), 0)
                    results[i]["rerank_score"] = float(s) + boost
            except Exception as e:
                logger.warning("KB rerank failed, 用向量分数排序: %s", e)
            results.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        final = results[:top_k]

        # ── Expand adjacent chunks (KB only) ──
        for r in final:
            if r.get("source") != "kb":
                continue
            fid = r.get("faiss_id", 0)
            prev_fid = fid - 1
            prev = next((c for c in chunk_meta if c.get("faiss_id") == prev_fid
                         and c.get("section_id") == r.get("section_id")), None)
            if prev:
                r["adjacent_prev"] = {
                    "chunk_id": prev.get("chunk_id"),
                    "chunk_type": prev.get("chunk_type"),
                    "text": prev.get("full_text", prev.get("text", ""))[:300],
                    "page_number": prev.get("page_number"),
                }
            next_fid = fid + 1
            nxt = next((c for c in chunk_meta if c.get("faiss_id") == next_fid
                        and c.get("section_id") == r.get("section_id")), None)
            if nxt:
                r["adjacent_next"] = {
                    "chunk_id": nxt.get("chunk_id"),
                    "chunk_type": nxt.get("chunk_type"),
                    "text": nxt.get("full_text", nxt.get("text", ""))[:300],
                    "page_number": nxt.get("page_number"),
                }

        logger.info("Global retrieval: %d results", len(final))
        return final

    @staticmethod
    def _extract_chapter_num(kp_id: str) -> int:
        """Extract chapter number from KP ID like 'D2-KP-3' → 3."""
        import re
        m = re.search(r'KP-(\d+)$', kp_id)
        return int(m.group(1)) if m else 0
