"""题目抽取器 — 从解析结果抽取题目（题解分离 + 章对应），写入独立试题库 TestQuestion。

与知识库共用同一「有序」原则：先把 result.json 的 raw_data 按 (页码, 纵坐标) 排成有序条目，
再在其上定位章标题 / 习题标题、按题号切题、拆分答案（状态机处理题解穿插与注解）。

章对应：章号精确匹配 → 语义匹配 → 位置回退（三级兜底，兼容无「第N章」前缀的学科）。
不做节（kp_id）对应：教材习题是章级命题，节归属本就不唯一，语义/关键词映射不可靠。
"""
import ast
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings

logger = logging.getLogger(__name__)

CHAPTER_RE = re.compile(r"第[一二三四五六七八九十\d]+章")
EXERCISE_RE = re.compile(r"^\s*(习题|练习|补充题|复习题|总习题)")
# 只认「1. / 1、 / 1．」这类顶层题号，不认「1) (1) 1）」等子项（避免把答案里的子步骤切成新题）
ITEM_RE = re.compile(r"^\s*(\d{1,3})[\.\、．]\s*\S")
# 目录条目带尾部页码（「…… 1」「 29」），据此区分目录与正文标题
_TRAILING_PAGE_RE = re.compile(r"[……\s]+\d+\s*$")
# 「三、」「四、」等序号前缀（答案书的「三、习题提示与解答」等节标题）
_SECTION_PREFIX_RE = re.compile(r"^[一二三四五六七八九十]+[、.．]\s*")
# 题干开头的题号「1. / 1、 / 1．」，抽取后应去掉（子题「1)」保留）
_STRIP_ITEM_RE = re.compile(r"^\s*\d{1,3}\s*[\.\、．]\s*")

_CN = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

CHAPTER_THRESHOLD = 0.35   # 章级语义匹配阈值（回退用）

# 抽题逻辑版本：任何影响题干/答案抽取结果的代码改动都要 +1。
# 增量构建时按此版本判断旧题是否需要重新抽取+校验（代码改变后之前的要重新跑）。
EXTRACTOR_VERSION = 9


def _cn_to_int(s: str) -> Optional[int]:
    """中文数字 → int（支持 一~十九、二十~九十九）。"""
    if s.isdigit():
        return int(s)
    if s in _CN:
        return _CN[s]
    if s.startswith("十"):
        return 10 + (_CN.get(s[1:], 0) if len(s) > 1 else 0)
    if "十" in s:
        a, b = s.split("十", 1)
        return _CN.get(a, 0) * 10 + (_CN.get(b, 0) if b else 0)
    return None


def _chapter_num(text: str) -> Optional[int]:
    """从「第N章」文本提取章号 N。"""
    m = CHAPTER_RE.search(text or "")
    if not m:
        return None
    return _cn_to_int(m.group(0)[1:-1])  # 去掉「第」「章」


def _ordered_items(doc_id: int) -> list[dict]:
    """读 result.json，返回按 (页码, 纵坐标) 排序的条目列表。"""
    result_file = Path(settings.DATA_DIR) / "parsed" / str(doc_id) / "result.json"
    if not result_file.exists():
        return []
    with open(result_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    raw_data = data.get("raw_data", [])

    def _y(bbox):
        try:
            arr = ast.literal_eval(bbox) if isinstance(bbox, str) else (bbox or [])
            return float(arr[1]) if len(arr) > 1 else 0.0
        except Exception:
            return 0.0

    items = []
    for it in raw_data:
        if not isinstance(it, dict):
            continue
        text = (it.get("text") or "").strip()
        if not text:
            continue
        try:
            page_idx = int(it.get("page_idx", 0))
        except (TypeError, ValueError):
            page_idx = 0
        items.append({
            "page": page_idx + 1,
            "y": _y(it.get("bbox")),
            "type": it.get("type", ""),
            "text": text,
        })
    items.sort(key=lambda x: (x["page"], x["y"]))
    return items


def _chapter_ranges(items: list[dict]) -> list[dict]:
    """按正文章标题首现页码，把文档切成章段 [(num, title, start, end)]。

    正文章标题可能是 text（章首页标题）或 header（页眉里第一次出现的章名），
    均取「第N章」开头、非目录（无尾部页码）的首现页码；end = 下一章 start。
    """
    seen: dict[int, dict] = {}
    for it in items:
        if it["type"] not in ("text", "title", "header"):
            continue
        t = (it["text"] or "").strip()
        if not CHAPTER_RE.match(t) or _TRAILING_PAGE_RE.search(t):
            continue
        num = _chapter_num(t)
        if num is None or num in seen:
            continue
        seen[num] = {"num": num, "title": t, "start": it["page"]}
    ordered = [seen[n] for n in sorted(seen)]
    for i in range(len(ordered)):
        ordered[i]["end"] = ordered[i + 1]["start"] if i + 1 < len(ordered) else None
    return ordered


def _is_exercise_heading(text: str) -> bool:
    """判断文本是否为习题节标题（习题/练习/补充题/复习题/总习题），可带「三、」等序号前缀。

    答案书结构：每章「一、内容提要 / 二、学习指导 / 三、习题提示与解答 / 四、补充题提示与解答」，
    只有「习题/补充题提示与解答」是题目+解答，「学习指导」是解析，需排除。
    """
    t = (text or "").strip()
    if not t or _TRAILING_PAGE_RE.search(t):
        return False
    t = _SECTION_PREFIX_RE.sub("", t, count=1)
    return bool(EXERCISE_RE.match(t))


def _is_learning_section_heading(text: str) -> bool:
    """「内容提要 / 学习指导」是非习题节，作为收集边界（其中是解析/复习内容，不是题目）。"""
    t = (text or "").strip()
    if not t or _TRAILING_PAGE_RE.search(t):
        return False
    t = _SECTION_PREFIX_RE.sub("", t, count=1)
    return t.startswith(("内容提要", "学习指导"))


def _find_exercise_sections(seg: list[dict]) -> list[tuple[str, int]]:
    """在一个章段内找出习题节：习题/补充题/总习题标题之后、到下一标题之前的文本。

    遇到「内容提要/学习指导」标题即停止收集，避免把解析内容当成题目。
    """
    sections: list[tuple[str, int]] = []
    cur_texts: list[str] | None = None
    cur_page = 0
    for it in seg:
        t = (it["text"] or "").strip()
        if it["type"] in ("text", "title") and _is_exercise_heading(t):
            if cur_texts is not None:
                sections.append(("\n".join(cur_texts), cur_page))
            cur_texts = []
            cur_page = it["page"]
            continue
        if it["type"] in ("text", "title") and _is_learning_section_heading(t):
            if cur_texts is not None:
                sections.append(("\n".join(cur_texts), cur_page))
            cur_texts = None
            continue
        if cur_texts is not None:
            if it["type"] not in ("header", "footer", "page_number", "page_footnote"):
                cur_texts.append(it["text"])
    if cur_texts is not None:
        sections.append(("\n".join(cur_texts), cur_page))
    return sections


def _is_question(q: str, a: Optional[str]) -> bool:
    """无习题标题的章节（答案书）里，判定编号条目是不是题目：只认带答案（解/答/证明）的。"""
    return bool(a)


_ANSWER_MARKER_RE = re.compile(r"^\s*(解|答)\s*[:：]?\s*")
_PROOF_ANSWER_RE = re.compile(r"^\s*(证明|证)\s+(?![:：])")   # 「证明 内容」=答案；「证明：内容」=题干
_SUB_Q_RE = re.compile(r"^\s*\d+\s*[\)）]\s*\S")              # 子题「1) …」
_ANNOTATION_RE = re.compile(r"^\s*(注意|说明|评注|注)\s*[:：]?")


def _split_answer(text: str) -> tuple[str, Optional[str]]:
    """题解分离（状态机 + 后向判断）：处理「题解穿插」「注解」「答案内部序号」。

    区分「子题 N)」与「答案里的步骤/方程序号」：
      子题后面还会再出现「解/证明/答」（子题与解交替），
      答案步骤后面不再有答案标记（同一道题的解答内部编号），
    故用「该行之后是否还有答案行」判断 N) 归属。
    """
    if not text:
        return "", None
    lines = text.split("\n")
    n = len(lines)
    is_answer = [bool(_ANSWER_MARKER_RE.match(l.strip()) or _PROOF_ANSWER_RE.match(l.strip())) for l in lines]
    is_subq = [bool(_SUB_Q_RE.match(l.strip())) for l in lines]
    # answer_after[i] = 第 i 行（不含）之后是否还有答案行
    answer_after = [False] * (n + 1)
    for i in range(n - 1, -1, -1):
        answer_after[i] = is_answer[i] or answer_after[i + 1]

    q_parts: list[str] = []
    a_parts: list[str] = []
    in_answer = False
    in_annotation = False
    for i, line in enumerate(lines):
        s = line.strip()
        if not in_annotation and _ANNOTATION_RE.match(s):
            in_annotation = True
            in_answer = False
            continue
        if in_annotation:
            # 注解结束于下一个「解/答」或子题「N)」；「证明」可能是注解内证明，不视为结束
            if _ANSWER_MARKER_RE.match(s) or is_subq[i]:
                in_annotation = False
            else:
                continue
        if is_subq[i]:
            if answer_after[i + 1]:
                # 后面还有「解/证明/答」→ 是子题，归题干
                q_parts.append(line)
                in_answer = False
            else:
                # 后面没有答案标记 → 是答案里的步骤序号，归答案
                a_parts.append(line)
                in_answer = True
        elif is_answer[i]:
            a_parts.append(line)
            in_answer = True
        else:
            if in_answer:
                a_parts.append(line)
            else:
                q_parts.append(line)
    q = "\n".join(q_parts).strip()
    a = "\n".join(a_parts).strip()
    return q, (_fix_array_hline(a) if a else None)


def _guess_type(text: str) -> str:
    """判断题型（映射到 TestQuestion 的 choice/fill/short_answer）。

    选择题需有明确选项结构（≥3 个行首 A/B/C/D 选项）或「选择题/选出」标记；
    「选择」单独出现常是「选择 i 与 k」这类数学指令，不视为选择题。
    「A.」也可能是「度量矩阵是 A.」这类句号，需行首 + 多选项才认。
    """
    if "选择题" in text or "选出" in text:
        return "choice"
    if len(re.findall(r"(?:^|\n)\s*[A-D][\.、\)]\s*\S", text)) >= 3:
        return "choice"
    if "填空" in text or "___" in text:
        return "fill"
    return "short_answer"


def _strip_item_number(text: str) -> str:
    """去掉题干开头的题号（1. / 1、 / 1．），子题「1)」保留。"""
    return _STRIP_ITEM_RE.sub("", text or "", count=1)


def _fix_array_hline(text: str) -> str:
    """修正 OCR 数组列错位：hline 被误夹在两个 & 之间（空单元格 + hline + 值）。

    MinerU 有时把 hline 当作一个单元格塞进「& hline &」，使该行 & 数量比列声明
    多一列，KaTeX 渲染错位。这里把 hline 提到行首、去掉前面多余的空单元格，
    例如「\\ & hline & r_3(x)=0」→「\\ hline & r_3(x)=0」。
    """
    if not text:
        return text
    BS = chr(92)  # 反斜杠
    pat = re.compile(
        re.escape(BS + BS) + r"[ \t]*&[ \t]*" + re.escape(BS + "hline") + r"[ \t]*&"
    )
    new_row = BS + BS + " " + BS + "hline &"
    return pat.sub(lambda _m: new_row, text)


def _unify_latex_delims(text: str) -> str:
    """统一 LaTeX 定界符：\\(...\\)→$...$、\\[...\\]→$$...$$，用于比较答案是否真有改动。

    避免 LLM 只把 $...$ 改成 \\(...\\) 这类等价写法时被误判为「有改动」。
    """
    if not text:
        return text
    BS = chr(92)
    t = re.sub(
        re.escape(BS + "[") + r"(.*?)" + re.escape(BS + "]"),
        lambda m: "$$\n" + m.group(1).strip() + "\n$$",
        text, flags=re.DOTALL,
    )
    t = re.sub(
        re.escape(BS + "(") + r"(.*?)" + re.escape(BS + ")"),
        lambda m: "$" + m.group(1).strip() + "$",
        t, flags=re.DOTALL,
    )
    return t


def _fix_orphan_delims(text: str) -> str:
    """去掉 LLM/OCR 残留的孤立 $$ 定界符（奇数个 $$ 时去掉最后一个，通常是尾部多打的）。"""
    if not text:
        return text
    if text.count("$$") % 2 == 1:
        idx = text.rfind("$$")
        if idx != -1:
            text = text[:idx] + text[idx + 2:]
    return text


def _split_questions(ex_text: str) -> list[str]:
    """把习题节的文本按题号（1. / 1、 / 1)）切成单个题目，并去掉题干开头的题号。"""
    groups: list[str] = []
    current: list[str] = []
    for line in ex_text.strip().split("\n"):
        ls = line.strip()
        if ITEM_RE.match(ls):
            if current:
                groups.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append("\n".join(current))
    return [_strip_item_number(g).strip() for g in groups if g.strip()]


def _match_chapter(heading: str, chapters: list, ch_embs, embedding_model):
    """章标题 → 知识树章节点（章号精确 → 语义 → 位置）。"""
    if not chapters:
        return None
    num = _chapter_num(heading)
    if num is not None:
        for c in chapters:
            if _chapter_num(c.title) == num:
                return c
        for c in chapters:
            if c.id.endswith(f"-{num}"):
                return c
    if ch_embs is not None and embedding_model is not None:
        q = embedding_model.encode([heading], normalize_embeddings=True)
        sims = (q @ ch_embs.T)[0]
        k = int(sims.argmax())
        if float(sims[k]) >= CHAPTER_THRESHOLD:
            return chapters[k]
    if num is not None and 1 <= num <= len(chapters):
        return chapters[num - 1]
    return None


def _resolve_chapter_override(override: str, chapters: list):
    """「指定章」：按章标题或节点 id 精确匹配；匹配不上返回 None（仍按原字符串存章名）。"""
    if not override:
        return None
    for c in chapters:
        if (c.title or c.id) == override or c.id == override:
            return c
    return None


async def extract_questions_to_bank(
    db: AsyncSession,
    doc_id: int,
    subject_id: int,
    embedding_model=None,
    chapter_override: str | None = None,
) -> dict:
    """抽取某文档在某学科下的题目到 TestQuestion（先删该文档旧的自动题，再重插）。

    chapter_override：可选「指定章」，整批题归入该章，跳过章自动识别（无结构纯题目列表时用）。
    """
    from app.models import KnowledgePoint, TestQuestion
    from app.services.rag_service import RAGService

    chapters = (await db.execute(
        select(KnowledgePoint).where(
            KnowledgePoint.subject_id == subject_id, KnowledgePoint.level == 0
        ).order_by(KnowledgePoint.sort_order, KnowledgePoint.id)
    )).scalars().all()

    items = _ordered_items(doc_id)
    if not items:
        return {"doc_id": doc_id, "subject_id": subject_id, "extracted": 0, "reason": "无解析条目"}

    if embedding_model is None:
        try:
            embedding_model = RAGService().embedding_model
        except Exception as e:
            logger.warning("无法加载嵌入模型，节映射改用关键词回退: %s", e)
            embedding_model = None

    ch_titles = [c.title or c.id for c in chapters]
    ch_embs = None
    if embedding_model is not None and ch_titles:
        ch_embs = embedding_model.encode(ch_titles, normalize_embeddings=True)

    # ── 章分段 → 章内找习题节 → 切题 + 题解分离 ──
    ranges = _chapter_ranges(items)
    extracted: list[dict] = []

    def _push(ch_node, q, a, page):
        title = ch_node.title if ch_node else None
        cid = ch_node.id if ch_node else None
        if chapter_override:
            node = _resolve_chapter_override(chapter_override, chapters)
            title = node.title if node else chapter_override
            cid = node.id if node else None
        extracted.append({
            "chapter_id": cid,
            "chapter_title": title,
            "question": q,
            "answer": a,
            "qtype": _guess_type(q),
            "page": page,
        })

    if ranges:
        for rng in ranges:
            ch_node = _match_chapter(rng["title"], chapters, ch_embs, embedding_model)
            seg = [it for it in items if it["page"] >= rng["start"] and (rng["end"] is None or it["page"] < rng["end"])]
            sections = _find_exercise_sections(seg)
            if sections:
                for sec_text, sec_page in sections:
                    for qtext in _split_questions(sec_text):
                        q, a = _split_answer(qtext)
                        if q and len(q) >= 3:
                            _push(ch_node, q, a, sec_page)
            else:
                # 无「习题」标题（如答案书）：按题号切，仅保留带答案或含求解动词的条目
                seg_text = "\n".join(it["text"] for it in seg)
                page = seg[0]["page"] if seg else None
                for qtext in _split_questions(seg_text):
                    q, a = _split_answer(qtext)
                    if q and len(q) >= 3 and _is_question(q, a):
                        _push(ch_node, q, a, page)
    else:
        # 无「第N章」标题：先按习题节抽取；若连习题节都没有（纯题目列表），整篇按题号切，仅保留带答案的条目
        sections = _find_exercise_sections(items)
        if sections:
            for sec_text, sec_page in sections:
                for qtext in _split_questions(sec_text):
                    q, a = _split_answer(qtext)
                    if not q or len(q) < 3:
                        continue
                    _push(_match_chapter(q, chapters, ch_embs, embedding_model), q, a, sec_page)
        else:
            # 无结构兜底：整篇按题号切题，只认带答案（解/答/证明）的条目，章归属语义匹配 + 可选指定章
            seg_text = "\n".join(
                it["text"] for it in items
                if it["type"] not in ("header", "footer", "page_number", "page_footnote")
            )
            page = items[0]["page"] if items else None
            for qtext in _split_questions(seg_text):
                q, a = _split_answer(qtext)
                if q and len(q) >= 3 and _is_question(q, a):
                    _push(_match_chapter(q, chapters, ch_embs, embedding_model), q, a, page)

    if not extracted:
        return {"doc_id": doc_id, "subject_id": subject_id, "extracted": 0, "reason": "未识别到题目"}

    # ── 增量入库：题干一致且已通过 LLM 校验的旧自动题保留（不重复校验）；
    #    未校验过的（旧题/新题）都会跑一次 LLM 校验并就地更新。 ──
    existing_rows = (await db.execute(
        select(TestQuestion).where(
            TestQuestion.source_doc_id == doc_id,
            TestQuestion.subject_id == subject_id,
            TestQuestion.source == "auto",
        )
    )).scalars().all()
    existing_by_key: dict[str, TestQuestion] = {}
    for e in existing_rows:
        existing_by_key.setdefault((e.question_text or "").strip(), e)

    # LLM 答案校验（可选，settings.LLM_VERIFY_ANSWERS 关闭可加速批量抽取）
    llm = None
    if settings.LLM_VERIFY_ANSWERS:
        try:
            from app.services.llm_service import LLMService
            llm = LLMService()
        except Exception as e:
            logger.warning("LLM 服务不可用，跳过答案校验: %s", e)
            llm = None

    seen_keys: set[str] = set()
    created = kept = updated = corrected = 0
    for i, o in enumerate(extracted):
        q_text = o["question"][:2000]
        key = q_text.strip()
        seen_keys.add(key)
        prev = existing_by_key.get(key)

        # 题干一致、已校验过、且抽取代码版本一致 → 保留（含已有 LLM 修正结果），跳过重复校验
        if prev is not None and prev.llm_verified and prev.extract_version == EXTRACTOR_VERSION:
            kept += 1
            continue

        # 需校验：先取抽取出的原始答案，再跑 LLM
        answer_text = o["answer"] or ""
        original_answer = None
        llm_corrected = False
        llm_verified = False
        if llm is not None and answer_text:
            try:
                llm_answer = await asyncio.to_thread(llm.verify_answer, o["question"], answer_text)
                if llm_answer:
                    llm_answer = _fix_orphan_delims(_unify_latex_delims(llm_answer))[:4000]
                    llm_verified = True
                    if llm_answer.strip() != _unify_latex_delims(answer_text).strip():
                        original_answer = answer_text
                        llm_corrected = True
                        corrected += 1
                    answer_text = llm_answer
            except Exception as ex:
                logger.warning("答案校验异常，保留原答案: %s", ex)

        if prev is not None:
            # 就地更新（题干不变、答案重校验、版本刷新）
            prev.chapter = o["chapter_title"]
            prev.question_type = o["qtype"]
            prev.answer_text = answer_text
            prev.original_answer = original_answer
            prev.llm_corrected = llm_corrected
            prev.llm_verified = llm_verified
            prev.extract_version = EXTRACTOR_VERSION
            prev.page_number = o["page"]
            updated += 1
        else:
            db.add(TestQuestion(
                subject_id=subject_id,
                chapter=o["chapter_title"],
                kp_id=None,
                question_type=o["qtype"],
                question_text=q_text,
                options=None,
                answer_text=answer_text,
                original_answer=original_answer,
                llm_corrected=llm_corrected,
                llm_verified=llm_verified,
                extract_version=EXTRACTOR_VERSION,
                difficulty=3,
                images=None,
                source="auto",
                source_doc_id=doc_id,
                page_number=o["page"],
                created_by=None,
                verified=False,
            ))
            created += 1

    # 删除本次未再出现的旧自动题
    for e in existing_rows:
        if (e.question_text or "").strip() not in seen_keys:
            await db.delete(e)

    await db.flush()
    total = created + updated + kept
    logger.info("📝 抽题: doc=%d subject=%d → 新增 %d / 更新 %d / 保留 %d 题（LLM 校验%s，修正 %d 题）",
                doc_id, subject_id, created, updated, kept,
                "开启" if llm else "关闭", corrected)
    return {
        "doc_id": doc_id, "subject_id": subject_id,
        "extracted": total, "created": created, "updated": updated, "kept": kept,
        "corrected": corrected, "llm": bool(llm),
    }
