"""Document Processor — MinerU CLI PDF parsing with progress tracking."""
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pymupdf as fitz

from app.config import settings

logger = logging.getLogger(__name__)

MINERU_TIMEOUT = int(os.getenv("MINERU_TIMEOUT", "7200"))


def _resolve_mineru_bin() -> Optional[str]:
    """定位 mineru CLI（必须跑在 MinerU 独立环境，避免 transformers 版本冲突）。

    版本冲突背景：后端 venv 用 transformers 5.x（sentence-transformers 需要），
    MinerU VLM 用 transformers 4.x（Qwen2VLConfig 有 max_position_embeddings）。
    故 MinerU 一律用独立环境运行（本地=系统 Python；服务器=独立 mineru venv）。
    """
    # 1. 显式配置（服务器部署时设成 mineru venv 的 mineru）
    if settings.MINERU_BIN:
        return settings.MINERU_BIN
    # 2. console script（自动绑定其所属环境）
    for name in ("mineru.exe", "mineru"):
        p = shutil.which(name)
        if p:
            return p
    # 3. 常见系统路径（本地 Windows）
    for cand in (r"E:\Python313\Scripts\mineru.exe", r"C:\Python313\Scripts\mineru.exe"):
        if os.path.exists(cand):
            return cand
    return None


MINERU_BIN = _resolve_mineru_bin()


def check_mineru_available() -> bool:
    """Check if MinerU CLI is available."""
    if not MINERU_BIN:
        return False
    try:
        result = subprocess.run(
            [MINERU_BIN, "--version"], capture_output=True, text=True, timeout=15
        )
        return result.returncode == 0
    except Exception:
        return False


MINERU_AVAILABLE = check_mineru_available()
if MINERU_AVAILABLE:
    logger.info("✅ MinerU CLI 可用: %s", MINERU_BIN)
else:
    logger.warning("⚠️ MinerU CLI 不可用，PDF 解析将失败（检查 MINERU_BIN 或 PATH）")


def _flatten_v2_content(content) -> str:
    """把 MinerU v2 的 content 字段（str 或嵌套 dict）展平成文本/LaTeX。

    v2 结构示例: {'paragraph_content': [
        {'type': 'text', 'content': '...'},
        {'type': 'equation_inline', 'content': 'g(x)'},
        {'type': 'equation_interline', 'content': '...'},
    ]}
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        parts = []
        for val in content.values():
            if isinstance(val, list):
                for seg in val:
                    if isinstance(seg, dict):
                        stype = seg.get("type", "")
                        stext = _flatten_v2_content(seg.get("content"))
                        if stype == "equation_inline":
                            parts.append(f"${stext}$")
                        elif stype in ("equation_interline", "equation", "formula"):
                            parts.append(f"$${stext}$$")
                        else:
                            parts.append(stext)
                    else:
                        parts.append(_flatten_v2_content(seg))
            elif isinstance(val, str):
                parts.append(val)
        return "".join(parts)
    return str(content)


def _normalize_raw_data(raw_data) -> List[Dict]:
    """把 MinerU 输出归一化成扁平 list[dict]（含 page_idx/text/type）。

    v2 格式: list[page] = list[item]，item 用 content(dict/str)；
    旧格式: 扁平 list[item]，item 用 page_idx/text。统一成后者。
    """
    if not isinstance(raw_data, list) or not raw_data:
        return []
    if isinstance(raw_data[0], list):
        flat = []
        for page_idx, page_items in enumerate(raw_data):
            for item in page_items:
                if isinstance(item, dict):
                    flat.append({
                        "page_idx": page_idx,
                        "type": item.get("type", ""),
                        "text": _flatten_v2_content(item.get("content")),
                        "bbox": item.get("bbox"),
                    })
        return flat
    return raw_data


class DocumentProcessor:
    """Process PDFs with MinerU high-quality mode."""

    def __init__(self):
        self.data_dir = Path(settings.DATA_DIR)
        self.parsed_dir = self.data_dir / "parsed"
        self.pdf_dir = self.data_dir / "pdfs"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    def process_pdf(self, pdf_path: Path, doc_id: int) -> Dict[str, Any]:
        """Process a PDF with MinerU high-quality mode, returning structured result."""
        result = {
            "doc_id": doc_id,
            "total_pages": 0,
            "pages": [],
            "regions": [],
            "tables": [],
            "formulas": [],
            "mineru_used": False,
        }

        if not MINERU_AVAILABLE:
            raise Exception("MinerU CLI 不可用")

        logger.info("🚀 MinerU 高精度解析: %s (doc_id=%d)", pdf_path, doc_id)
        mineru_result = self._run_mineru(pdf_path, doc_id)
        if mineru_result:
            result.update(mineru_result)
            result["mineru_used"] = True
            return result
        raise Exception("MinerU 解析返回空结果")

    def _run_mineru(self, pdf_path: Path, doc_id: int) -> Dict[str, Any]:
        """Run MinerU via CLI 子进程（连常驻 mineru-api，hybrid-engine 高精度）。

        子进程跑在独立 MinerU 环境（transformers 4.x），不 import mineru 到后端，
        避免与后端 venv 的 transformers 5.x 冲突。
        """
        doc_dir = self.parsed_dir / str(doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        # 总页数
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()
        logger.info("📄 PDF 共 %d 页", total_pages)

        log_file = doc_dir / "mineru.log"
        with open(log_file, "w", encoding="utf-8", errors="ignore", buffering=1) as log_f:
            def _write_log(msg: str):
                import datetime
                ts = datetime.datetime.now().strftime("%H:%M:%S")
                log_f.write(f"[{ts}] {msg}\n")
                log_f.flush()

            _write_log(f"开始解析: {pdf_path.name} ({total_pages} 页)")
            _write_log("模式: hybrid-engine + OCR + formula + table, effort=high（高精度，含 VLM）")

            cmd = [
                MINERU_BIN, "-p", str(pdf_path), "-o", str(doc_dir),
                "-b", "hybrid-engine",
                "-m", "ocr",
                "-l", "ch",
                "--effort", "high",
                "--image-analysis", "true",
            ]
            # 优先连常驻 mineru-api
            api_url = settings.MINERU_API_URL
            if api_url:
                cmd += ["--api-url", api_url]
                _write_log(f"连接常驻 MinerU API: {api_url}")

            logger.info("🚀 MinerU CLI: %s", " ".join(cmd))
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=MINERU_TIMEOUT,
                encoding="utf-8", errors="ignore",
            )

            # 常驻服务失败 → 回退临时服务
            if result.returncode != 0 and api_url:
                _write_log(f"常驻服务失败，回退临时服务: {result.stderr[:800]}")
                fallback = [c for c in cmd if c != "--api-url" and c != api_url]
                result = subprocess.run(
                    fallback, capture_output=True, text=True, timeout=MINERU_TIMEOUT,
                    encoding="utf-8", errors="ignore",
                )

            if result.stdout:
                log_f.write(result.stdout)
            if result.stderr:
                log_f.write(result.stderr)
            log_f.flush()

            if result.returncode != 0:
                raise Exception(f"MinerU 解析失败: {result.stderr[:500]}")

            _write_log("✅ MinerU 解析完成，正在收集输出文件...")

        # 找输出（md + content_list json）
        md_files = sorted(doc_dir.rglob("*.md"), key=lambda f: f.stat().st_size, reverse=True)
        json_files = list(doc_dir.rglob("*.json"))
        content_json = None
        for pattern in ["content_list_v2", "content_list", "middle", "model"]:
            for f in json_files:
                if pattern in f.name:
                    content_json = f
                    break
            if content_json:
                break
        if content_json is None and json_files:
            content_json = sorted(json_files, key=lambda f: f.stat().st_size, reverse=True)[0]

        if content_json:
            logger.info("✅ MinerU output: %s", content_json)
            with open(content_json, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            return self._convert_result(raw_data, doc_id, total_pages)

        if md_files:
            logger.info("✅ Markdown output: %s", md_files[0])
            return self._convert_md(md_files[0], doc_id)

        all_files = list(doc_dir.rglob("*"))
        raise Exception(f"未找到输出文件: {[str(f.relative_to(doc_dir)) for f in all_files[:20]]}")

    def _convert_result(self, raw_data: Union[Dict, List], doc_id: int, total_pages: int) -> Dict[str, Any]:
        """Convert MinerU JSON output to canonical format and save result.json."""
        raw_data = _normalize_raw_data(raw_data)
        result = {
            "doc_id": doc_id,
            "total_pages": total_pages,
            "pages": [],
            "raw_data": raw_data,
            "mineru_used": True,
            "markdown": "",
        }

        if raw_data:
            max_page = max(
                (it.get("page_idx", 0) for it in raw_data if isinstance(it, dict)),
                default=0,
            )
            result["total_pages"] = max(total_pages, max_page + 1)

        # Read markdown if exists
        doc_dir = self.parsed_dir / str(doc_id)
        md_files = list(doc_dir.rglob("*.md"))
        if md_files:
            try:
                with open(md_files[0], "r", encoding="utf-8") as f:
                    result["markdown"] = f.read()
            except Exception as e:
                logger.warning("读取 Markdown 失败: %s", e)

        result_file = doc_dir / "result.json"
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def _convert_md(self, md_path: Path, doc_id: int) -> Dict[str, Any]:
        """Convert Markdown-only output to canonical format."""
        with open(md_path, "r", encoding="utf-8") as f:
            content = f.read()

        result = {
            "doc_id": doc_id,
            "total_pages": 1,
            "pages": [{"page_num": 1, "regions": [{"region_type": "text", "content": content}]}],
            "mineru_used": True,
            "markdown": content,
        }

        result_file = self.parsed_dir / str(doc_id) / "result.json"
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return result

    def get_parsed_result(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """Read cached parsed result."""
        result_file = self.parsed_dir / str(doc_id) / "result.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None
