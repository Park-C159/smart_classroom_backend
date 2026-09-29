"""Knowledge Tree Builder — reads MinerU content_list_v2.json, uses TOC for structure.

Hierarchy: Chapter (level 0) → Section (level 1) → Knowledge Point (level 2)
Each KP has its own content blocks; each block tracks its PDF page number.
"""
import json
import logging
import re
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import KnowledgePoint, ContentChunk, Document

logger = logging.getLogger(__name__)

BLOCK_PATTERNS = [
    ("definition", re.compile(r'(?:^|。|；)\s*(?:定义|Definition)\s*\d*')),
    ("theorem", re.compile(r'(?:^|。|；)\s*(?:定理|Theorem|命题|Proposition|引理|Lemma|推论|Corollary)\s*\d*')),
    ("example", re.compile(r'(?:^|。|；)\s*(?:例|Example|例题)\s*\d*')),
    ("proof", re.compile(r'(?:^|。|；)\s*(?:证明|Proof|证)[:：]')),
    ("remark", re.compile(r'(?:^|。|；)\s*(?:注|注意|Remark|说明)[:：]')),
]

# Fallback: match type keywords anywhere in first 40 chars
_TYPE_KEYWORDS = {
    "definition": r'定义|Definition',
    "theorem": r'定理|Theorem|命题|Proposition|引理|Lemma|推论|Corollary',
    "example": r'例\d|Example|例题',
    "proof": r'证明|Proof',
    "remark": r'注|注意|Remark|说明',
}

CHAPTER_RE = re.compile(r'^\*?\s*第[一二三四五六七八九十\d]+章\s*')  # Allow optional * prefix (e.g. *第八章)
SECTION_RE = re.compile(r'^§\s*\d+')
ALT_SECTION_RE = re.compile(r'^[一二三四五六七八九十]+[、]')  # Chinese-numbered: 一、二、三、

# Strip trailing page numbers like "…… 1", "… 34", " 8"
_TRAILING_PAGE_RE = re.compile(r'[……\s]+\d+\s*$')


def _clean_title(text: str) -> str:
    """Remove trailing page numbers (e.g. '第一章 多项式 …… 1' → '第一章 多项式')."""
    return _TRAILING_PAGE_RE.sub('', text).strip()


def _extract_page_num(text: str) -> int:
    """Extract trailing page number from a TOC entry."""
    m = re.search(r'[……\s]+(\d+)\s*$', text)
    return int(m.group(1)) if m else 0


def _extract_text(content_obj: dict) -> str:
    """Extract plain text from MinerU nested content structure."""
    if not content_obj:
        return ""
    if isinstance(content_obj, list):
        return "".join(_extract_text(seg) for seg in content_obj)
    if isinstance(content_obj, dict):
        t = content_obj.get("type", "")
        c = content_obj.get("content", "")
        if t == "text":
            return c
        if t in ("equation", "equation_inline"):
            return f"${c}$"
        if t == "equation_interline":
            return f"$$\n{c}\n$$"
        if content_obj.get("math_content"):
            return f"$$\n{content_obj['math_content']}\n$$"
        for key in ("title_content", "paragraph_content", "page_header_content",
                     "page_number_content", "page_footer_content", "item_content",
                     "list_items", "table_cells"):
            val = content_obj.get(key)
            if val:
                if key == "list_items":
                    parts = []
                    for li in val:
                        item_text = _extract_text(li.get("item_content", []))
                        item_text = re.sub(r'^\s*\d+[\)\.\、]?\s*', '', item_text, count=1)
                        if item_text.strip():
                            parts.append(item_text)
                    return "\n".join(parts)
                elif key == "table_cells":
                    rows = []
                    for cell in val:
                        rows.append(_extract_text(cell))
                    return " | ".join(rows)
                else:
                    return _extract_text(val)
    return ""


def _is_toc_level_heading(text: str) -> bool:
    """Check if a heading text is a TOC-level section (not a sub-section)."""
    if re.match(r'^[一二三四五六七八九十]+[、]', text): return True  # 一、二、三、
    if SECTION_RE.match(text): return True     # §1, §2
    if re.match(r'^(?:习题|练习|补充题|总习题)', text): return True
    return False


_BLOCK_TYPE_PATTERNS = [
    ("example",    re.compile(r'^例(\s*\d+|\s+\S)')),
    ("definition", re.compile(r'^定义\s*\d*')),
    ("theorem",    re.compile(r'^(?:定理|命题|引理|推论)\s*\d*')),
    ("proof",      re.compile(r'^(?:证明|证)[:：\s]')),
    ("remark",     re.compile(r'^(?:注|注意|说明)[:：]')),
]


def _classify_block(text: str) -> str:
    """Classify a text paragraph by its first line pattern."""
    first = text.strip().split('\n')[0][:60]
    for bt, pat in _BLOCK_TYPE_PATTERNS:
        if pat.search(first):
            return bt
    stripped = re.sub(r'^\s*\d+[\.\、\s]+', '', first)
    for bt, pat in _BLOCK_TYPE_PATTERNS:
        if bt == "example" and pat.search(stripped):
            return bt
    return "text"


class KnowledgeTreeService:
    """Build knowledge tree from MinerU content_list_v2.json using TOC as blueprint."""

    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR)

    async def build_from_content_list(
        self, doc_id: int, subject_id: int, db: AsyncSession, is_primary: bool = True
    ) -> dict[str, Any]:
        content_list_path = self._find_content_list(doc_id)
        if not content_list_path:
            raise FileNotFoundError(f"找不到 doc {doc_id} 的 content_list_v2.json")

        with open(content_list_path, "r", encoding="utf-8") as f:
            pages = json.load(f)

        # Non-primary docs get their own KP tree with a doc prefix
        doc_type = "textbook" if is_primary else "reference"
        structure = self._parse_structure(pages, doc_id=doc_id if not is_primary else 0, is_primary=is_primary, doc_type=doc_type)

        # 知识树已改为手动构建（每个学科一棵可展开目录，叶节点=知识点，不含内容块）。
        # 这里只从解析结果生成内容块（知识库，供答疑检索），不再自动写章节/知识点树。
        chunk_count = await self._store_content_chunks(db, structure, doc_id, subject_id)

        # 教材解析不再抽取习题/题库、也不用 VLM 处理图片：习题功能已移除，题库后续用新方式构建
        return {
            "doc_id": doc_id, "is_primary": is_primary,
            "chapters": len(structure["chapters"]),
            "content_chunks": chunk_count,
        }

    # ── File location ──

    def _find_content_list(self, doc_id: int) -> Path | None:
        doc_dir = self.data_dir / "parsed" / str(doc_id)
        if not doc_dir.exists():
            return None
        # Try direct path first (content_list_v2.json in a subdirectory)
        import os as _os
        for entry in _os.listdir(str(doc_dir)):
            sub_path = doc_dir / entry
            if sub_path.is_dir():
                candidate = sub_path / "content_list_v2.json"
                if candidate.exists():
                    return candidate
        # Fallback to rglob
        matches = list(doc_dir.rglob("*content_list_v2*.json"))
        return matches[0] if matches else None

    # ── Structure parsing (markdown + V2 hybrid) ──

    def _find_markdown(self, doc_id: int):
        """Find the markdown file for a document."""
        import os as _os
        from pathlib import Path as _Path
        doc_dir = self.data_dir / 'parsed' / str(doc_id)
        for entry in _os.listdir(str(doc_dir)):
            sub = _Path(doc_dir) / entry
            if not sub.is_dir():
                continue
            hybrid = sub / 'hybrid_ocr'
            search = hybrid if hybrid.exists() else sub
            for f in _os.listdir(str(search)):
                if f.endswith('.md'):
                    return search / f
        return None

    def _parse_structure(self, pages: list, doc_id: int = 0, is_primary: bool = True, doc_type: str = "textbook") -> dict:
        """Parse TOC from markdown + body titles from V2 + extract content from markdown."""
        kp_prefix = f'D{doc_id}-' if doc_id > 0 else ''

        md_path = self._find_markdown(doc_id if doc_id > 0 else 1)
        if not md_path:
            return {'chapters': [], 'doc_type': 'textbook'}
        with open(md_path, 'r', encoding='utf-8') as f:
            md_lines = f.readlines()

        # Step 1: TOC from markdown
        toc_entries = self._parse_md_toc(md_lines)

        # Step 2: Body titles from V2, matched to markdown lines
        body_titles = self._get_body_titles_from_v2(pages)
        self._match_v2_to_md(body_titles, md_lines)

        # Step 3: Sequential TOC → body titles
        section_titles = [h for h in body_titles if not h.get('is_chapter')]
        ch_toc = {}
        for ch_title, sec_title in toc_entries:
            ch_toc.setdefault(ch_title, []).append(sec_title)

        heading_idx = 0
        chapter_idx = 0
        chapters = []

        for ch_title, sec_titles in ch_toc.items():
            chapter_idx += 1
            ch_id = f'{kp_prefix}KP-{chapter_idx}'
            chapter = {'title': ch_title, 'kp_id': ch_id, 'sections': []}

            for sec_title in sec_titles:
                if heading_idx < len(section_titles):
                    h = section_titles[heading_idx]
                    chapter['sections'].append({
                        'title': sec_title, 'kps': [], 'exercises': [],
                        '_page': h.get('page', 0),
                        '_md_line': h.get('md_line', 0),
                    })
                    heading_idx += 1
            chapters.append(chapter)

        # Step 4: Set section end lines
        all_sec = []
        for ch in chapters:
            all_sec.extend(ch['sections'])
        for i in range(len(all_sec) - 1):
            all_sec[i]['_end_line'] = all_sec[i + 1]['_md_line']
        if heading_idx < len(section_titles):
            all_sec[-1]['_end_line'] = section_titles[heading_idx]['md_line']
        else:
            all_sec[-1]['_end_line'] = len(md_lines)

        # Step 5: Extract & classify content
        for sec in all_sec:
            if sec['_md_line'] == 0:
                continue
            raw_text = self._extract_md_section(md_lines, sec['_md_line'], sec['_end_line'])
            is_ex = bool(re.search(r'习题|补充题|练习', sec['title']))

            if is_ex:
                if doc_type == "reference":
                    blocks, exercises = self._chunk_reference_exercises(raw_text, sec['_page'])
                else:
                    blocks, exercises = self._chunk_exercises(raw_text, sec['_page'])
            else:
                blocks = self._parse_content_to_blocks(raw_text, sec['_page'])
                exercises = []

            sec['kps'] = [
                {'title': b['content'][:60].replace('\n', ' ').strip(),
                 'blocks': [{'type': b['type'], 'content': b['content'], 'page_number': b.get('page')}],
                 'images': b.get('images', [])}
                for b in blocks
            ]
            sec['exercises'] = exercises

        # Clean up
        for ch in chapters:
            for sec in ch['sections']:
                sec.pop('_page', None); sec.pop('_md_line', None); sec.pop('_end_line', None)
        return {'chapters': chapters, 'doc_type': 'textbook'}

    # ── TOC from markdown ──

    def _parse_md_toc(self, md_lines: list) -> list:
        entries = []; cur_ch = None; in_toc = False
        ch_re = re.compile(r'^##[\s\\*]*第[一二三四五六七八九十\d]+章')
        sec_re = re.compile(r'^[一二三四五六七八九十]+[、]|^§\s*\d+|^(?:习题|练习|补充题|总习题)')
        for line in md_lines:
            s = line.strip()
            if s == '## 目录' or (ch_re.match(s) and _TRAILING_PAGE_RE.search(s) and not in_toc):
                in_toc = True
                if s != '## 目录': cur_ch = _clean_title(s[3:])
                continue
            if not in_toc: continue
            if ch_re.match(s) and not _TRAILING_PAGE_RE.search(s): break
            if s.startswith('# ') and CHAPTER_RE.search(s): break
            if ch_re.match(s) and _TRAILING_PAGE_RE.search(s): cur_ch = _clean_title(s[3:])
            elif sec_re.match(s) and cur_ch: entries.append((cur_ch, _clean_title(s)))
        return entries

    # ── Body titles from V2 ──

    def _get_body_titles_from_v2(self, pages: list) -> list:
        body_start = 4
        for pg_idx, page in enumerate(pages):
            for block in page:
                if block['type'] == 'title' and block['content'].get('level', 0) in (1, 2):
                    text = _extract_text(block['content'].get('title_content', []))
                    if CHAPTER_RE.search(text) and not _TRAILING_PAGE_RE.search(text):
                        body_start = pg_idx; break
            else: continue
            break

        headings = []
        for pg_idx, page in enumerate(pages):
            if pg_idx < body_start: continue
            for block in page:
                if block['type'] != 'title': continue
                level = block['content'].get('level', 0)
                if level not in (1, 2): continue
                text = _extract_text(block['content'].get('title_content', []))
                if not text or len(text) < 2: continue
                is_ch = bool(CHAPTER_RE.search(text))
                is_section = _is_toc_level_heading(text)
                if not is_ch and not is_section: continue
                if _TRAILING_PAGE_RE.search(text): continue
                headings.append({'title': text, 'page': pg_idx + 1, 'is_chapter': is_ch, 'md_line': 0})
        return headings

    def _match_v2_to_md(self, headings: list, md_lines: list) -> None:
        search_from = 0
        for h in headings:
            h_title = h['title'].replace(' ', '')
            for i in range(search_from, len(md_lines)):
                s = md_lines[i].strip()
                if not (s.startswith('## ') or s.startswith('# ')): continue
                title = s.lstrip('#').strip().replace(' ', '')
                if title == h_title or (len(h_title) > 4 and (title[:8] == h_title[:8] or h_title[:6] in title)):
                    h['md_line'] = i; search_from = i + 1; break

    # ── Content extraction ──

    def _extract_md_section(self, md_lines: list, start_line: int, end_line: int) -> str:
        """Extract section content, stopping at next ## or # heading to prevent boundary leaks."""
        heading_re = re.compile(r'^#{1,2}\s+')
        parts = []
        stopped_early = False
        for i in range(start_line + 1, end_line):
            line = md_lines[i].rstrip()
            # Stop at next section/chapter heading (safety boundary)
            if heading_re.match(line) and i > start_line + 2:
                stopped_early = True
                break
            parts.append(line)
        return '\n'.join(parts).strip()

    def _parse_content_to_blocks(self, raw_text: str, page_hint: int = 0) -> list:
        """Split only at KP boundaries — no blank-line splitting.
        Everything between two KP markers (定义/定理/证明/例/注) stays as one block."""
        _KP_BOUNDARY = re.compile(
            r'^\s*(定义\s*\d*|定理\s*\d*|命题\s*\d*|引理\s*\d*|推论\s*\d*|'
            r'证明|证\s*[:：]|'
            r'例\s*\d+|例\s+\S|'
            r'注\s*[:：]|注意\s*[:：]|说明\s*[:：])'
        )

        lines = raw_text.strip().split('\n')
        groups = []
        current = []
        for line in lines:
            stripped = line.strip()
            if _KP_BOUNDARY.match(stripped):
                if current:
                    groups.append('\n'.join(current))
                current = [line]
            else:
                current.append(line)
        if current:
            groups.append('\n'.join(current))

        blocks = []
        for text in groups:
            text = text.strip()
            if not text:
                continue
            if text.startswith('![') and text.endswith(')'):
                m = re.match(r'!\[(.*?)\]\((.*?)\)', text)
                if m:
                    blocks.append({'type': 'image_block', 'content': text, 'page': page_hint, 'images': [m.group(2)]})
                continue
            if text.startswith('<table>'):
                blocks.append({'type': 'table', 'content': text, 'page': page_hint, 'images': []})
                continue
            btype = _classify_block(text)
            img_paths = re.findall(r'!\[.*?\]\((.*?)\)', text)
            clean = re.sub(r'!\[.*?\]\(.*?\)', '', text).strip()
            if not clean or len(clean) < 2:
                continue
            blocks.append({'type': btype, 'content': clean, 'page': page_hint, 'images': img_paths})
        return blocks

    def _chunk_reference_exercises(self, raw_text: str, page_hint: int = 0) -> tuple:
        """Chunk reference exercises: split by top-level 'N. ' numbering only.
        Sub-items (1), a), etc. won't match — they stay within the same exercise."""
        blocks = []
        exercises = []
        lines = raw_text.strip().split('\n')
        current_lines = []
        current_started = False

        for line in lines:
            stripped = line.strip()
            # Only match "N. " (number + period + space) — not "1)", "(1)", "1、"
            is_top_level = bool(re.match(r'^\s*\d{1,3}\.\s+\S', stripped))

            if is_top_level and current_started:
                text = '\n'.join(current_lines).strip()
                cleaned = re.sub(r'^\s*\d+\.\s*', '', text, count=1)
                if len(cleaned) > 3:
                    exercises.append({'text': cleaned, 'page': page_hint})
                    blocks.append({'type': 'exercise', 'content': text, 'page': page_hint})
                current_lines = [line]

            elif is_top_level and not current_started:
                current_lines = [line]
                current_started = True

            elif current_started:
                current_lines.append(line)

            else:
                if stripped:
                    current_lines = [line]
                    current_started = True

        if current_lines:
            text = '\n'.join(current_lines).strip()
            cleaned = re.sub(r'^\s*\d+\.\s*', '', text, count=1)
            if len(cleaned) > 3:
                exercises.append({'text': cleaned, 'page': page_hint})
                blocks.append({'type': 'exercise', 'content': text, 'page': page_hint})

        return blocks, exercises

    def _chunk_exercises(self, raw_text: str, page_hint: int = 0) -> tuple:
        """Chunk textbook exercises: each numbered item is one exercise, no merging."""
        blocks = []
        exercises = []
        paragraphs = re.split(r'\n\n+', raw_text.strip())
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            lines = para.split('\n')
            sub_groups = []
            current = []
            for line in lines:
                ls = line.strip()
                is_ex_start = bool(re.match(r'^\s*\d+[\.\、]\s*[^0-9]', ls))
                if is_ex_start and current:
                    sub_groups.append(('\n'.join(current), True))
                    current = [line]
                else:
                    current.append(line)
            if current:
                first_is_ex = bool(re.match(r'^\s*\d+[\.\、]\s*[^0-9]', current[0].strip()))
                sub_groups.append(('\n'.join(current), first_is_ex))
            for sub_text, is_new_ex in sub_groups:
                sub_text = sub_text.strip()
                if not sub_text:
                    continue
                img_paths = re.findall(r'!\[.*?\]\((.*?)\)', sub_text)
                sub_clean = re.sub(r'!\[.*?\]\(.*?\)', '', sub_text).strip()
                if not sub_clean or len(sub_clean) < 2:
                    if img_paths:
                        blocks.append({'type': 'image_block', 'content': sub_text, 'page': page_hint, 'images': img_paths})
                    continue
                # Each numbered item = one exercise, strip the number
                et = re.sub(r'^\s*\d+[\.\)\、）]\s*', '', sub_clean, count=1)
                if et and len(et) > 3:
                    exercises.append({'text': et, 'page': page_hint})
                blocks.append({'type': 'exercise', 'content': sub_text, 'page': page_hint, 'images': img_paths})
        return blocks, exercises

    # ── VLM image processing ──

    async def _process_vlm_images(self, structure: dict, doc_id: int) -> int:
        """Process images in all blocks with VLM, inject descriptions inline.

        For each block that has image paths, resolves the actual image file,
        calls VLM to get a description, and injects `[📷 VLM: description]`
        right after the image markdown in the block content.
        Returns the number of images processed.
        """
        # Collect all image paths across all blocks
        image_tasks = []  # [(kp_dict, block_dict, img_path)]
        for chapter in structure.get("chapters", []):
            for section in chapter.get("sections", []):
                for kp in section.get("kps", []):
                    for block in kp.get("blocks", []):
                        for img_path in block.get("images", []):
                            image_tasks.append((kp, block, img_path))

        if not image_tasks:
            return 0

        # Resolve image paths and collect unique ones
        doc_dir = self.data_dir / "parsed" / str(doc_id)
        resolved = {}  # img_path → full_filesystem_path or None

        for _, _, img_path in image_tasks:
            if img_path in resolved:
                continue
            full_path = self._resolve_image_file(img_path, doc_dir)
            resolved[img_path] = full_path

        # Skip if no images could be resolved
        resolved_count = sum(1 for v in resolved.values() if v)
        if resolved_count == 0:
            logger.info("📷 VLM: 0/%d images resolved, skipping", len(resolved))
            return 0

        # Call VLM for each unique image
        try:
            from app.services.vlm_service import VLMService
            vlm = VLMService()
        except Exception as e:
            logger.warning("📷 VLM service unavailable: %s", e)
            return 0

        vlm_results = {}  # img_path → description
        for img_path, full_path in resolved.items():
            if not full_path:
                vlm_results[img_path] = "[VLM: image file not found]"
                continue
            try:
                desc = vlm.describe(str(full_path))
                vlm_results[img_path] = desc or "[VLM: no description]"
                logger.info("📷 VLM: %s → %s", img_path, desc[:60] if desc else "empty")
            except Exception as e:
                logger.warning("📷 VLM error for %s: %s", img_path, e)
                vlm_results[img_path] = f"[VLM error: {e}]"

        # Inject descriptions into block content at image positions
        processed = 0
        for kp, block, img_path in image_tasks:
            desc = vlm_results.get(img_path)
            if not desc:
                continue
            # Store in images metadata
            for img_entry in kp.get("images", []):
                if isinstance(img_entry, dict) and img_entry.get("path") == img_path:
                    img_entry["vlm_desc"] = desc
                    break
            # Inject into block content at image position
            vlm_tag = f"\n[📷 VLM: {desc}]"
            # Try exact markdown replacement
            if f"![]({img_path})" in block["content"]:
                block["content"] = block["content"].replace(
                    f"![]({img_path})", f"![]({img_path}){vlm_tag}", 1)
            else:
                # Regex match for any alt text
                pat = re.compile(rf'!\[.*?\]\({re.escape(img_path)}\)')
                if pat.search(block["content"]):
                    block["content"] = pat.sub(
                        lambda m: f"{m.group()}{vlm_tag}", block["content"], count=1)
            processed += 1

        logger.info("📷 VLM: %d images processed across %d blocks", processed, len(image_tasks))
        return processed

    def _resolve_image_file(self, img_path: str, doc_dir: Path) -> Path | None:
        """Resolve a relative image path to an actual file on disk."""
        import os as _os
        filename = img_path.split("/")[-1] if "/" in img_path else img_path
        for root, dirs, files in _os.walk(str(doc_dir)):
            if filename in files:
                return Path(root) / filename
        return None

    # ── DB storage ──

    async def _store_knowledge_points(
        self, db: AsyncSession, structure: dict, subject_id: int
    ) -> int:
        """Store 3-level hierarchy: chapter (0) → section (1) → KP (2)."""
        count = 0
        for chapter in structure["chapters"]:
            ch_id = chapter["kp_id"]
            existing = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == ch_id))
            if existing:
                existing.title = chapter["title"]
                existing.subject_id = subject_id
            else:
                db.add(KnowledgePoint(
                    id=ch_id, title=chapter["title"],
                    chapter=chapter["title"],
                    level=0, sort_order=count,
                    subject_id=subject_id,
                ))
            count += 1

            for si, section in enumerate(chapter.get("sections", [])):
                sec_id = f"{ch_id}.{si + 1}"
                sec_title = section.get("title", "")
                existing = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == sec_id))
                if existing:
                    existing.title = sec_title
                    existing.parent_id = ch_id
                    existing.level = 1
                    existing.subject_id = subject_id
                else:
                    db.add(KnowledgePoint(
                        id=sec_id, title=sec_title,
                        chapter=chapter["title"],
                        parent_id=ch_id, level=1,
                        sort_order=count,
                        subject_id=subject_id,
                    ))
                count += 1

                for ki, kp in enumerate(section.get("kps", [])):
                    kp_id = f"{sec_id}.{ki + 1}"
                    kp_title = kp.get("title", "")
                    existing = await db.scalar(select(KnowledgePoint).where(KnowledgePoint.id == kp_id))
                    if existing:
                        existing.title = kp_title
                        existing.parent_id = sec_id
                        existing.level = 2
                        existing.subject_id = subject_id
                    else:
                        db.add(KnowledgePoint(
                            id=kp_id, title=kp_title,
                            chapter=chapter["title"],
                            parent_id=sec_id, level=2,
                            sort_order=count,
                            subject_id=subject_id,
                        ))
                    count += 1

        await db.flush()
        logger.info("📊 知识树: %d KPs (subject=%d, 3 levels)", count, subject_id)
        return count

    async def _store_content_chunks(
        self, db: AsyncSession, structure: dict, doc_id: int, subject_id: int = None
    ) -> int:
        """Store content chunks (knowledge base) with PDF page numbers and VLM descriptions."""
        count = 0
        for chapter in structure["chapters"]:
            for si, section in enumerate(chapter.get("sections", [])):
                sec_id = f"{chapter['kp_id']}.{si + 1}"
                for ki, kp in enumerate(section.get("kps", [])):
                    kp_id = f"{sec_id}.{ki + 1}"
                    for block in kp.get("blocks", []):
                        # Build images JSON with VLM descriptions
                        images_data = None
                        block_images = kp.get("images", [])
                        if block_images:
                            images_data = [
                                {"path": img.get("path", img) if isinstance(img, dict) else img,
                                 "vlm_desc": img.get("vlm_desc") if isinstance(img, dict) else None}
                                for img in block_images
                            ]
                        db.add(ContentChunk(
                            kp_id=kp_id,
                            chunk_type=block.get("type", "text"),
                            content=block.get("content", "")[:5000],
                            page_number=block.get("page_number"),
                            subject_id=subject_id,
                            source_doc_id=doc_id,
                            images=images_data,
                        ))
                        count += 1
        await db.flush()
        logger.info("📊 知识库: %d 块 (doc=%d)", count, doc_id)
        return count

    # ── Preview ──

    def preview_structure(self, doc_id: int) -> dict:
        content_list_path = self._find_content_list(doc_id)
        if not content_list_path:
            return {"error": f"找不到 doc {doc_id} 的 content_list_v2.json"}
        with open(content_list_path, "r", encoding="utf-8") as f:
            pages = json.load(f)
        return self._parse_structure(pages)

    # ── KP Summaries ──

    async def generate_kp_summaries(self, db: AsyncSession, llm_service=None, force: bool = False) -> list:
        """Generate LLM summaries for level-2 knowledge points.

        Summaries are concise Chinese descriptions that preserve LaTeX math notation.
        Only processes level-2 KPs (the actual knowledge points within sections).
        """
        kps = (await db.execute(
            select(KnowledgePoint).where(KnowledgePoint.level == 2)
        )).scalars().all()
        results = []
        for kp in kps:
            if kp.summary and not force:
                continue
            # Get first 3 content chunks for better context
            chunks = (await db.execute(
                select(ContentChunk).where(ContentChunk.kp_id == kp.id).limit(3)
            )).scalars().all()
            context = "\n".join(c.content[:300] for c in chunks) if chunks else kp.title
            if llm_service:
                try:
                    summary = ""
                    for _, text in llm_service.get_stream_response(
                        query=f"知识点上下文：\n{context}",
                        context=None,
                        system_prompt=(
                            "你是一名数学教材编辑专家。请根据以下知识点内容，提炼出该知识点的核心概念名（8-15字）。"
                            "要求：1.保留LaTeX数学公式（如$f(x)$、$\\mathbb{R}$等）"
                            "2.用简洁的数学术语命名，如'多项式的整除定义'、'最大公因式的性质'"
                            "3.只返回知识点名称，不要其他解释"
                        ),
                    ):
                        summary += text
                    kp.summary = summary.strip()
                    results.append({"kp_id": kp.id, "summary": kp.summary})
                except Exception as e:
                    logger.warning("KP摘要失败 %s: %s", kp.id, e)
            else:
                # No LLM available: use the existing title, trimmed
                kp.summary = kp.title[:20]
                results.append({"kp_id": kp.id, "summary": kp.summary})
        await db.flush()
        logger.info("📊 已生成 %d 个KP摘要", len(results))
        return results
