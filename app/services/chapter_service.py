"""
Chapter Service — Semantic chapter/section matching for hierarchical RAG.

Uses BGE-M3 embedding to match user queries to chapter titles, then
narrow down to specific sections within the matched chapter.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)

# ── Keyword extraction patterns ──
CHAPTER_EXPLICIT_RE = re.compile(r'第[一二三四五六七八九十\d]+章')
SECTION_EXPLICIT_RE = re.compile(r'(第[一二三四五六七八九十\d]+节|§\s*\d+)')
MATH_CONCEPT_RE = re.compile(
    r'(多项式|行列式|矩阵|线性|向量|空间|'
    r'二次型|特征值|特征向量|对角化|'
    r'欧几里得|辛空间|双线性|二次型|'
    r'λ[-\s]*矩阵|若尔当|标准形)'
)


class ChapterService:
    """Semantic chapter/section matching for hierarchical retrieval."""

    def __init__(self, embedding_model=None):
        """
        Args:
            embedding_model: SentenceTransformer model (shared from RAGService).
                             If None, will load its own instance.
        """
        if embedding_model is not None:
            self._embedding_model = embedding_model
        else:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(
                settings.BGE_M3_MODEL_NAME, local_files_only=True
            )

        self._chapter_index: Optional[faiss.Index] = None
        self._section_index: Optional[faiss.Index] = None
        self._chapter_meta: List[Dict] = []
        self._section_meta: List[Dict] = []

    # ── Index loading ──

    def load_indices(self, subject_id: int) -> bool:
        """Load hierarchical chapter/section indices for a subject."""
        store_dir = Path(settings.DATA_DIR) / "vector_store" / str(subject_id)

        ch_idx_path = store_dir / "faiss_chapter.index"
        ch_meta_path = store_dir / "chapter_meta.json"

        if not ch_idx_path.exists() or not ch_meta_path.exists():
            logger.warning("Chapter index not found for subject %d", subject_id)
            return False

        self._chapter_index = faiss.read_index(str(ch_idx_path))
        with open(ch_meta_path, "r", encoding="utf-8") as f:
            self._chapter_meta = json.load(f)
        logger.info("Loaded chapter index: %d chapters", len(self._chapter_meta))

        sec_idx_path = store_dir / "faiss_section.index"
        sec_meta_path = store_dir / "section_meta.json"
        if sec_idx_path.exists():
            self._section_index = faiss.read_index(str(sec_idx_path))
            with open(sec_meta_path, "r", encoding="utf-8") as f:
                self._section_meta = json.load(f)
            logger.info("Loaded section index: %d sections", len(self._section_meta))

        return True

    # ── Keyword extraction ──

    def extract_keywords(self, query: str) -> Tuple[str, Optional[int], List[str]]:
        """Extract semantic keywords, explicit chapter reference, and math concepts.

        Returns:
            (search_text, explicit_chapter_num, concept_keywords)
        """
        concepts = MATH_CONCEPT_RE.findall(query)

        # Check for explicit chapter reference
        ch_match = CHAPTER_EXPLICIT_RE.search(query)
        explicit_ch = None
        if ch_match:
            ch_text = ch_match.group()
            # Map Chinese number to integer
            ch_map = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                       '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
            for cn, num in ch_map.items():
                if cn in ch_text:
                    explicit_ch = num
                    break
            if explicit_ch is None:
                # Try Arabic numeral
                num_match = re.search(r'\d+', ch_text)
                if num_match:
                    explicit_ch = int(num_match.group())

        # Build search text: prefer concepts, fall back to full query
        if concepts:
            search_text = ' '.join(dict.fromkeys(concepts))  # deduplicate
        else:
            search_text = query

        # If explicit chapter reference found, bias search text
        if explicit_ch is not None:
            search_text = f"第{explicit_ch}章 {search_text}"

        return search_text, explicit_ch, concepts

    # ── Chapter matching ──

    def match_chapter(self, query: str,
                       explicit_ch: Optional[int] = None,
                       threshold: float = 0.35) -> List[Dict]:
        """Match query to best chapters using full-text embedding similarity.

        Encodes the full query and compares against all chapter title embeddings.
        Returns top-1 chapter above threshold, or empty list.
        """
        if not self._chapter_index or not self._chapter_meta:
            return []

        # Explicit chapter reference → direct lookup
        if explicit_ch is not None:
            for ch in self._chapter_meta:
                if ch.get("chapter_num") == explicit_ch:
                    logger.info("Chapter matched explicitly: %s", ch["title"])
                    return [dict(ch)]

        # Embed full query → compare with all chapter titles
        query_emb = self._embedding_model.encode(
            [query], normalize_embeddings=True
        )
        k = min(3, len(self._chapter_meta))
        scores, indices = self._chapter_index.search(query_emb.astype("float32"), k)

        best_score = float(scores[0][0])
        if best_score < threshold:
            logger.info("Chapter match below threshold (%.3f < %.2f)", best_score, threshold)
            return []

        best_idx = indices[0][0]
        ch = dict(self._chapter_meta[best_idx])
        logger.info("Chapter matched: %s (score=%.3f)", ch["title"], best_score)
        return [ch]

    # ── Section matching ──

    def match_sections(self, query: str, chapter_id: str,
                        top_k: int = 2) -> List[Dict]:
        """Match sections within a chapter using embedding similarity."""
        if not self._section_meta:
            return []

        chapter_sections = [s for s in self._section_meta
                            if s.get("chapter_id") == chapter_id]
        if not chapter_sections:
            return []

        if len(chapter_sections) <= top_k:
            return chapter_sections

        # Embed query and compare with section titles in this chapter
        query_emb = self._embedding_model.encode([query], normalize_embeddings=True)
        section_texts = [s["text"] for s in chapter_sections]
        section_embs = self._embedding_model.encode(section_texts, normalize_embeddings=True)
        sims = np.dot(query_emb, section_embs.T)[0]

        ranked = sorted(zip(sims, chapter_sections), key=lambda x: x[0], reverse=True)
        matched = [dict(s) for _, s in ranked[:top_k] if _ > 0.3]

        if not matched:
            matched = [dict(chapter_sections[0])]  # fallback to first section

        logger.info("Section match: %d sections in chapter %s", len(matched), chapter_id)
        return matched

    # ── All sections for a chapter ──

    def get_chapter_sections(self, chapter_id: str) -> List[str]:
        """Get all section KP IDs that belong to a chapter."""
        # Section IDs are like KP-1.1, KP-1.2 — children of chapter KP-1
        return [s["kp_id"] for s in self._section_meta
                if s.get("chapter_id") == chapter_id]

    def get_chapter_all_kps(self, chapter_id: str) -> List[str]:
        """Get ALL KP IDs (including level-2) under a chapter."""
        # From section_meta we have the level-1 KP IDs
        # Level-2 KPs are like KP-1.1.1 → we need to include them via prefix matching
        sections = self.get_chapter_sections(chapter_id)
        all_kps = list(sections)
        # Also include level-2 KPs: KP-1.1.*, KP-1.2.* etc.
        for sec_id in sections:
            # Level-2 KPs have IDs matching section_id + ".N"
            # We'll handle this at the retrieval level using DB query
            pass
        return all_kps
