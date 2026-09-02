"""LLM Service — OpenAI-compatible streaming API (DeepSeek / Qwen)."""
import logging
from typing import List, Dict, Generator, Optional
from openai import OpenAI

from app.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Singleton LLM client for streaming chat generation."""

    _instance: Optional["LLMService"] = None

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

    def get_sync_response(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3) -> str:
        """Non-streaming response for short tasks (topic generation, etc.)."""
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body={"thinking": {"type": "disabled"}},  # 短任务直接答，不开思考
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
            resp = self.client.chat.completions.create(
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
            system_prompt = (
                "你是高等代数答疑助手。请像一位熟悉教材的老师一样直接回答学生的问题。\n\n"
                "## 规则\n"
                "1. 使用 LaTeX 格式，行内用 $...$，独立公式用 $$...$$。\n"
                "2. 回答简洁，直接给答案。\n"
                "3. 绝对不要提[参考资料][检索到][教材中说][根据第X章]之类的话，也不要标注页码。"
            )

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
            messages.append({"role": "system", "content": f"以下教材内容供你参考，请直接用来回答问题，不要提及这些内容的来源：\n\n{context_text}"})

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
            stream = self.client.chat.completions.create(
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
            yield ("content", f"LLM 调用失败: {str(e)}")
