"""Web search service — 优先百度 AI 搜索 API，无 key 时回退 Bing HTML 抓取。

百度搜索 API：https://qianfan.baidubce.com/v2/ai_search/web_search
鉴权：Authorization: Bearer <API Key>（key 放在 .env 的 BAIDU_SEARCH_API_KEY）
"""
import logging
import re
import httpx
from bs4 import BeautifulSoup

from app.config import settings

logger = logging.getLogger(__name__)

# 疑问词前缀：搜索前去掉句首结构词（什么是/证明/求…），保留关键术语
_QUESTION_PREFIXES = [
    "什么是", "是什么", "什么叫", "如何", "怎么", "为什么", "请问",
    "试证", "求证", "证明", "求", "设", "已知", "判断", "计算",
    "简述", "说明", "解释", "求解", "化简", "讨论",
]


def _clean_query(query: str) -> str:
    """去掉句首疑问词和标点，保留关键术语，整体搜索。"""
    q = query.strip()
    while True:
        stripped = False
        for w in sorted(_QUESTION_PREFIXES, key=len, reverse=True):
            if q.startswith(w):
                q = q[len(w):].strip()
                stripped = True
                break
        if not stripped:
            break
    q = re.sub(r"[，。？！、；：,.?!;:]+", " ", q)
    return " ".join(q.split()) or query.strip()


class WebSearchService:
    """联网搜索。优先百度 API（混合中英文更准），失败回退 Bing。"""

    def __init__(self):
        self._client = httpx.Client(
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                ),
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
            timeout=15,
            follow_redirects=True,
        )

    def search(self, query: str, max_results: int = 5) -> list:
        """返回 [{"title", "url", "snippet"}]；失败返回空列表。"""
        if not query or not query.strip():
            return []
        query = _clean_query(query)
        logger.info("联网搜索（清洗后）: %s", query)

        if settings.BAIDU_SEARCH_API_KEY:
            results = self._baidu_search(query, max_results)
            if results:
                return results
            logger.warning("百度搜索无结果，回退 Bing")

        return self._bing_search(query, max_results)

    # ── 百度 AI 搜索 API ──

    def _baidu_search(self, query: str, max_results: int) -> list:
        try:
            resp = self._client.post(
                "https://qianfan.baidubce.com/v2/ai_search/web_search",
                headers={
                    "Authorization": f"Bearer {settings.BAIDU_SEARCH_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "messages": [{"role": "user", "content": query}],
                    "top_k": max_results,
                    "stream": False,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            # 实际返回结构：{"request_id": ..., "references": [{title, url, snippet, content, ...}]}
            items = data.get("references") or data.get("result") or []
            if isinstance(items, dict):
                items = items.get("references", [])
            results = []
            for item in items:
                title = item.get("title") or item.get("name") or ""
                url = item.get("url") or item.get("link") or ""
                snippet = item.get("snippet") or item.get("content") or item.get("abstract") or item.get("summary") or ""
                if url:
                    results.append({"title": title, "url": url, "snippet": snippet})
                if len(results) >= max_results:
                    break
            return results
        except Exception as e:
            logger.warning("百度搜索失败: %s", e)
            return []

    # ── Bing HTML 抓取（无 key 回退）──

    def _bing_search(self, query: str, max_results: int) -> list:
        # 纯中文单短语加引号精确匹配，避免被分词成单字
        if re.fullmatch(r"[一-鿿·\s]+", query):
            search_q = f'"{query}"'
        else:
            search_q = query
        try:
            resp = self._client.get(
                "https://www.bing.com/search",
                params={"q": search_q, "count": str(max_results)},
            )
            resp.raise_for_status()
            return self._parse(resp.text, max_results)
        except Exception as e:
            logger.warning("Bing 搜索失败: %s", e)
            return []

    def _parse(self, html: str, max_results: int) -> list:
        soup = BeautifulSoup(html, "lxml")
        results = []
        for li in soup.select("li.b_algo"):
            a = li.select_one("h2 a")
            if not a or not a.get("href"):
                continue
            title = a.get_text(" ", strip=True)
            url = a["href"]
            snippet_el = li.select_one(".b_caption p, p")
            snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
            results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= max_results:
                break
        return results


web_search_service = WebSearchService()
