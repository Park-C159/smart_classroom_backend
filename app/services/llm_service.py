"""LLM Service — OpenAI-compatible streaming API (DeepSeek / Qwen)."""
import logging
import os
import threading
import time
from typing import List, Dict, Generator, Optional
from openai import (
    OpenAI,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
)

from app.config import settings

logger = logging.getLogger(__name__)


def _is_rate_limited(e: Exception) -> bool:
    """判断是否为限流/瞬时错误（可安全重试）。

    DeepSeek 在请求过密时会返回 "Too much connections in one minute, Please try later"（429）。
    这里同时覆盖官方 SDK 异常类与文本特征，兼容中转/代理返回的裸文本。
    """
    if isinstance(e, (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)):
        return True
    msg = str(e).lower()
    return any(k in msg for k in (
        "429", "too much", "too many", "try later", "rate limit",
        "ratelimit", "overloaded", "service unavailable", "busy",
    ))


class LLMService:
    """Singleton LLM client for streaming chat generation."""

    _instance: Optional["LLMService"] = None

    # ── 全局限流：防止批量判分/摘要等循环里短时间打爆 DeepSeek 的连接数限制 ──
    # DeepSeek 限制「每分钟连接数」；这里按「最小请求间隔」串行化所有 LLM 请求，
    # 默认 55 次/分钟（留出余量，低于 66 上限）。可用环境变量 LLM_MAX_RPM 覆盖。
    _rate_lock = threading.Lock()
    _last_call_ts = 0.0
    MAX_RPM = int(os.getenv("LLM_MAX_RPM", "55"))
    MIN_INTERVAL = 60.0 / max(1, MAX_RPM)

    def __new__(cls) -> "LLMService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.api_key = settings.DEEPSEEK_API_KEY
        self.base_url = settings.DEEPSEEK_BASE_URL
        self.model = settings.DEEPSEEK_CHAT_MODEL

        if not self.api_key:
            logger.warning("DEEPSEEK_API_KEY not set — LLM calls will fail")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info("LLM 初始化: model=%s base_url=%s", self.model, self.base_url)

    def _wait_for_rate_slot(self) -> None:
        """全局限流：保证两次 LLM 请求之间至少间隔 MIN_INTERVAL 秒。

        锁内 sleep 会串行化所有线程的 LLM 请求（这正是目的：一次只发一个连接），
        使每分钟请求数不超过 DeepSeek 的连接数上限。
        """
        with LLMService._rate_lock:
            now = time.time()
            wait = LLMService._last_call_ts + LLMService.MIN_INTERVAL - now
            if wait > 0:
                time.sleep(wait)
            LLMService._last_call_ts = time.time()

    def _create_with_retry(self, retries: int = 5, base_delay: float = 2.0, **kwargs):
        """带指数退避重试的 chat.completions.create。

        针对 DeepSeek 限流（"Too much connections in one minute"）及瞬时网络错误，
        自动重试；非限流错误（如鉴权/参数错误）第一次就抛出，不浪费时间。
        """
        delay = base_delay
        last_err: Optional[Exception] = None
        for attempt in range(retries):
            try:
                self._wait_for_rate_slot()
                return self.client.chat.completions.create(**kwargs)
            except Exception as e:  # noqa: BLE001 — 统一按是否限流决定是否重试
                last_err = e
                if not _is_rate_limited(e) or attempt == retries - 1:
                    raise
                logger.warning(
                    "LLM 限流/瞬时错误，%.1fs 后重试（%d/%d）: %s",
                    delay, attempt + 1, retries, str(e)[:150],
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise last_err

    def grade_short_answer(self, question_text: str, answer_text: str, user_answer: str) -> dict | None:
        """对简答题（含证明题）用 LLM 参考标准答案自动判分。

        Returns: {"score": 0~1, "feedback": "评语"}；失败时返回 None（调用方回退精确匹配）。
        """
        import json

        prompt = f"""你是高等代数阅卷老师。请根据标准答案对学生的简答题作答评分。

题目：
{question_text}

标准答案：
{answer_text or "（无标准答案）"}

学生作答：
{user_answer or "（未作答）"}

请判断学生作答是否正确、完整，给出 0~1 的得分（0 完全错误，1 完全正确，可给小数）和一句简短评语。
严格只输出 JSON，不要输出其它内容：
{{"score": 0.8, "feedback": "评语"}}"""

        try:
            text = self.get_sync_response(prompt, max_tokens=400, temperature=0.1)
            s = text.find("{")
            e = text.rfind("}")
            if s != -1 and e != -1 and e > s:
                obj = json.loads(text[s:e + 1])
                score = float(obj.get("score", 0))
                return {
                    "score": max(0.0, min(1.0, score)),
                    "feedback": str(obj.get("feedback", ""))[:500],
                }
        except Exception as ex:
            logger.warning("简答判分失败: %s", ex)
        return None

    # 长答案分块校验：单次请求输出长度有限，答案较长时拆成多段分别请求 LLM，
    # 避免单次请求因上下文/输出长度受限而截断（多段校验后再按原顺序拼接）。
    VERIFY_CHUNK_CHARS = 1200

    def verify_answer(self, question_text: str, answer_text: str) -> str | None:
        """校验并修正教材解析出的题目答案（OCR/切分可能有误），返回修正后的答案文本；失败返回 None。

        答案较长时拆成多段分别请求 LLM，逐段修正后按原顺序拼接。
        """
        if not answer_text:
            return None
        chunks = self._split_verify_chunks(answer_text)
        if len(chunks) <= 1:
            return self._verify_answer_once(question_text, answer_text)
        parts: list[str] = []
        for ch in chunks:
            r = self._verify_answer_once(question_text, ch)
            parts.append(r if r else ch)  # 单段失败则保留原文
        return "\n\n".join(parts)

    def _split_verify_chunks(self, text: str) -> list[str]:
        """按行切分长答案，只在公式块（$$…$$、\\begin…\\end）之外断开，避免切断公式。"""
        if len(text) <= self.VERIFY_CHUNK_CHARS:
            return [text]
        lines = text.split("\n")
        chunks: list[str] = []
        cur: list[str] = []
        cur_len = 0
        in_display = False
        env_depth = 0
        for ln in lines:
            # 当前是否处于「两个公式块之间」的安全边界（上一行结束时不在任何块内）
            safe = (not in_display) and env_depth <= 0
            if cur and cur_len + len(ln) + 1 > self.VERIFY_CHUNK_CHARS and safe:
                chunks.append("\n".join(cur))
                cur, cur_len = [], 0
            cur.append(ln)
            cur_len += len(ln) + 1
            # 追加后再更新状态：$$ 成对切换 display 状态（奇数个 $$ 表示状态翻转）
            in_display = in_display != (ln.count("$$") % 2 == 1)
            env_depth += ln.count("\\begin{") - ln.count("\\end{")
        if cur:
            chunks.append("\n".join(cur))
        return chunks if len(chunks) > 1 else [text]

    def _verify_answer_once(self, question_text: str, answer_text: str) -> str | None:
        """对单段答案做一次 LLM 校验/修正。"""
        prompt = f"""你是一名严谨的数学教材校对老师。下面是从教材中解析出的一道题及其答案，答案可能因 OCR 或题解切分有误（多/漏字符、错符号、公式错位、表格列错位等）。

【题目】
{question_text}

【当前答案】
{answer_text or "（无）"}

请把答案整理成清晰、正确的参考答案：
- 修正错误：错符号、错数字、多/漏字符、公式错位等；不要新增推导/证明步骤，也不要改动数学内容与数值。
- 步骤清晰：保留并补全「解、所以、答」等必要的步骤标记，使推导过程一目了然。

LaTeX 排版要求：
- 行内公式统一用 $...$，独立公式用 $$...$$；所有 $ 必须成对出现，不得有多余或孤立的 $。
- 分数、上下标、花括号等命令结构完整、成对闭合。
- 数组、矩阵等表格：每行 & 数量与列声明一致；横线（hline）独占一行，不夹在两个 & 之间。
- 辗转相除法（欧几里得算法）的运算表：OCR 常把「商」与「多项式系数」挤在同一列导致列错位、对不齐。请重排成对齐正确的表格——商单独一列，被除式系数、除式系数各一列，各步之间用 hline 分隔；只重排列，不改动系数、商、余式的数值。

只输出答案本身，不要任何解释、前缀、引号或“修正后：”字样。"""
        try:
            return self.get_sync_response(prompt, max_tokens=2000, temperature=0.1).strip()
        except Exception as ex:
            logger.warning("答案校验失败: %s", ex)
            return None

    def get_sync_response(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3, enable_thinking: bool = False) -> str:
        """Non-streaming response for short tasks (topic generation, etc.).

        enable_thinking=True 时开启推理思考（先 reason 再答），适合需要推敲的数学答案校准。
        """
        try:
            resp = self._create_with_retry(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "enabled" if enable_thinking else "disabled"}},
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.error("LLM sync call failed: %s", e)
            raise

    def resolve_history(
        self, question: str, history: List[Dict]
    ) -> Dict:
        """Agent: determine if history is needed and extract relevant context.

        Returns: {needed: bool, context: str, rewritten_question: str}
        Uses a fast, low-token LLM call.
        """
        if not history:
            return {"needed": False, "context": "", "rewritten_question": question}

        # Build compact history summary
        hist_text = "\n".join(
            f"[{h['role']}]: {h['content'][:300]}"
            for h in history[-6:]  # last 3 rounds
        )

        prompt = f"""你是一个对话历史分析器。判断用户当前问题是否需要回顾对话历史，如果需要则提取相关上下文并改写问题。

## 对话历史
{hist_text}

## 当前问题
{question}

## 规则
1. 如果问题是独立的新问题（不需要历史就能理解），输出: NO
2. 如果需要历史（有指代词如"它"、"这个"、追问、延续之前话题），输出:
YES
相关上下文: <从历史中提取的1-3条关键信息>
改写问题: <融入历史上下文的完整问题>

请严格按照以下格式输出，不要多余内容：
NO
或
YES
相关上下文: ...
改写问题: ..."""

        try:
            resp = self._create_with_retry(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=300,
                extra_body={"thinking": {"type": "disabled"}},  # 历史分析直接答，不开思考
            )
            text = resp.choices[0].message.content.strip()

            if text.startswith("NO"):
                return {"needed": False, "context": "", "rewritten_question": question}

            context = ""
            rewritten = question
            for line in text.split("\n"):
                if line.startswith("相关上下文:") or line.startswith("相关上下文："):
                    context = line.split(":", 1)[-1].strip()
                elif line.startswith("改写问题:") or line.startswith("改写问题："):
                    rewritten = line.split(":", 1)[-1].strip()

            logger.info("History agent: needed=True, rewritten=%.50s", rewritten)
            return {"needed": True, "context": context, "rewritten_question": rewritten or question}
        except Exception as e:
            logger.warning("History agent failed: %s", e)
            return {"needed": False, "context": "", "rewritten_question": question}

    @staticmethod
    def build_subject_prompt(subject_name: Optional[str] = None) -> str:
        """按学科构建 system prompt，并强调优先参考 RAG 召回内容。"""
        name = subject_name or "数学"
        return (
            f"你是「{name}」学科的答疑助手，请像一位熟悉该学科教材的老师一样回答学生的问题。\n\n"
            "## 规则\n"
            "1. 优先基于提供的教材内容作答：只要教材内容与问题相关，就必须以教材内容为准，不要脱离教材自行发挥。\n"
            "2. 只有教材内容不足或与问题无关时，才可用自己的知识补充，并保持严谨、准确。\n"
            "3. 使用 LaTeX 格式：行内用 $...$，独立公式用 $$...$$。\n"
            "4. 回答简洁，直接给出答案和必要推导。\n"
            "5. 绝对不要出现[参考资料][检索到][教材中说][根据第X章]等字样，也不要标注页码或来源。"
        )

    def get_stream_response(
        self,
        query: str,
        context: List[Dict],
        system_prompt: Optional[str] = None,
        max_tokens: int = 16384,
        history: Optional[List[Dict]] = None,
        deep_think: bool = False,
    ) -> Generator[tuple, None, None]:
        """Streaming generation with optional context injection.

        产出 (kind, text) 元组：kind 为 "content"（正文）或 "thinking"（思考过程，
        仅 deep_think=True 时输出推理模型的 reasoning_content）。

        Args:
            query: User question.
            context: List of retrieved chunks, each with 'text' key.
            system_prompt: Optional custom system prompt.
            history: Previous conversation messages [{role, content}, ...].
            deep_think: 是否流式输出思考过程（reasoning_content）。
        """
        has_context = context is not None and len(context) > 0

        if system_prompt is None:
            system_prompt = self.build_subject_prompt()

        # Build context text — include adjacent chunks to prevent semantic breakage
        context_text = ""
        if has_context:
            parts = []
            for i, item in enumerate(context, 1):
                text = item.get("full_text", item.get("text", ""))

                # Prepend adjacent previous chunk (KB only, 300 chars)
                prev = item.get("adjacent_prev")
                if prev and prev.get("text"):
                    prev_text = prev["text"][:300]
                    text = prev_text + "\n\n" + text

                # Append adjacent next chunk (KB only, 300 chars)
                nxt = item.get("adjacent_next")
                if nxt and nxt.get("text"):
                    nxt_text = nxt["text"][:300]
                    text = text + "\n\n" + nxt_text

                # Cap total per chunk to avoid oversized context
                if len(text) > 2000:
                    text = text[:2000] + "..."

                # Include answer if this is a QB exercise match
                answer = item.get("answer_text", "")
                if answer:
                    if len(answer) > 1000:
                        answer = answer[:1000] + "..."
                    text = f"{text}\n参考答案：{answer}"

                parts.append(text)
            context_text = "\n\n---\n\n".join(parts)

        messages = [{"role": "system", "content": system_prompt}]

        if has_context and context_text:
            messages.append({"role": "system", "content": f"以下教材内容是权威依据，请优先严格依据它作答，不要提及来源：\n\n{context_text}"})

        # Insert conversation history (within token budget, newest first)
        if history:
            MAX_HISTORY_TOKENS = 8000
            hist_tokens = 0
            hist_messages = []
            for h in reversed(history):
                role = h.get("role", "user")
                if role not in ("user", "assistant"):
                    continue
                content = h.get("content", "")[:4000]
                est = len(content) * 0.4  # rough token estimate
                if hist_tokens + est > MAX_HISTORY_TOKENS:
                    break
                hist_tokens += est
                hist_messages.append({"role": role, "content": content})
            messages.extend(reversed(hist_messages))

        user_content = f"问题：{query}"
        if has_context:
            user_content += "\n\n请根据以上教材内容回答。不要提出处。"
        else:
            user_content += "\n\n请使用你的知识回答。"
        messages.append({"role": "user", "content": user_content})

        try:
            logger.info("🚀 调用 LLM 流式 API (has_context=%s, 资料数=%d, deep_think=%s)",
                        has_context, len(context) if context else 0, deep_think)
            stream = self._create_with_retry(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=max_tokens,
                stream=True,
                # deepseek-v4-pro 通过 thinking 参数开关思考模式（enabled=思考，disabled=直接答）
                extra_body={"thinking": {"type": "enabled" if deep_think else "disabled"}},
            )
            chunk_count = 0
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                # 深度思考：把推理模型的思考过程流式输出
                if deep_think and getattr(delta, "reasoning_content", None):
                    yield ("thinking", delta.reasoning_content)
                if delta.content:
                    chunk_count += 1
                    yield ("content", delta.content)
            logger.info("✅ LLM 流式完成，共 %d 个块", chunk_count)
        except Exception as e:
            logger.error("❌ LLM 调用失败: %s", e, exc_info=True)
            if _is_rate_limited(e):
                yield ("content", "LLM 服务暂时繁忙（请求过多），请稍等片刻后重试。")
            else:
                yield ("content", f"LLM 调用失败: {str(e)[:200]}")
