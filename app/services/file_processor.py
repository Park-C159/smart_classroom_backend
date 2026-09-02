"""File Processor — extract text from uploaded files using MinerU."""
import asyncio
import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

from fastapi import UploadFile

from app.config import settings

logger = logging.getLogger(__name__)


class FileProcessor:
    """Extract text from uploaded files via MinerU.

    两种后端：
    - `pipeline`  : 经典 OCR/文本流水线（Layout + OCR + 公式 + 表格），
                    轻量，文字 + LaTeX 公式精确。
    - `vlm-engine`: 本地 VLM（MinerU2.5-Pro-1.2B）语义识别，含图形理解，
                    显存较高但无需 API key。

    优先连接常驻 mineru-api 服务（settings.MINERU_API_URL），失败则回退
    每次起临时服务（冷启动会重载模型，较慢）。
    """

    def __init__(self):
        self.mineru_timeout = 300  # 5 minutes

    # ── 对外方法 ──

    async def extract_text(
        self,
        file: UploadFile,
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> str:
        """pipeline 后端：提取文字 + LaTeX 公式（无图形语义）。"""
        return await self._extract(
            file, backend="pipeline",
            enable_formula=enable_formula, enable_table=enable_table,
        )

    async def extract_image_text(self, file: UploadFile) -> str:
        """vlm-engine 后端：本地 VLM 语义识别图片（文字 + 公式 + 图形描述）。"""
        return await self._extract(file, backend="vlm-engine")

    # ── 内部 ──

    async def _extract(
        self,
        file: UploadFile,
        backend: str,
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> str:
        content = await file.read()
        suffix = Path(file.filename).suffix.lower() if file.filename else ".tmp"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        output_dir = tempfile.mkdtemp()
        try:
            return await asyncio.to_thread(
                self._run_mineru, tmp_path, output_dir, backend, enable_formula, enable_table
            )
        finally:
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass
            shutil.rmtree(output_dir, ignore_errors=True)

    def _run_mineru(
        self,
        file_path: str,
        output_dir: str,
        backend: str = "pipeline",
        enable_formula: bool = True,
        enable_table: bool = True,
    ) -> str:
        """执行 MinerU CLI，返回解析出的 markdown 文本。"""
        cmd = self._build_cmd(file_path, output_dir, backend, enable_formula, enable_table)

        # 优先连常驻 mineru-api 服务，失败回退临时服务
        if settings.MINERU_API_URL:
            try:
                return self._execute(cmd + ["--api-url", settings.MINERU_API_URL], output_dir)
            except Exception as e:
                logger.warning(
                    "常驻 mineru-api (%s) 调用失败，回退临时服务: %s",
                    settings.MINERU_API_URL, e,
                )
        return self._execute(cmd, output_dir)

    def _build_cmd(
        self,
        file_path: str,
        output_dir: str,
        backend: str,
        enable_formula: bool,
        enable_table: bool,
    ) -> list:
        cmd = ["mineru", "-p", file_path, "-o", output_dir, "-b", backend]
        if backend == "pipeline":
            cmd += [
                "-m", "auto",
                "-l", "ch",
                "--formula", "true" if enable_formula else "false",
                "--table", "true" if enable_table else "false",
            ]
        elif backend == "vlm-engine":
            cmd += ["--image-analysis", "true"]
        return cmd

    def _execute(self, cmd: list, output_dir: str) -> str:
        logger.info("🚀 MinerU 文件解析: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.mineru_timeout,
            encoding="utf-8", errors="ignore",
        )
        if result.returncode != 0:
            raise Exception(f"MinerU 解析失败: {result.stderr}")

        output_path = Path(output_dir)
        md_files = list(output_path.rglob("*.md"))
        if md_files:
            with open(md_files[0], "r", encoding="utf-8") as f:
                return f.read()

        json_files = list(output_path.rglob("*.json"))
        if json_files:
            with open(json_files[0], "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return "\n".join(item.get("text", "") for item in data if "text" in item)
                elif isinstance(data, dict) and "pages" in data:
                    return "\n".join(p.get("content", "") for p in data["pages"])

        raise Exception("未找到 MinerU 输出文件")
