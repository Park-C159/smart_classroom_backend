"""Speech-to-Text — local Whisper (faster-whisper), on-demand load/unload."""
import asyncio
import gc
import logging
import os
import tempfile
import threading
import time

from fastapi import UploadFile

from app.config import settings

logger = logging.getLogger(__name__)


class STTService:
    """Transcribe audio to text via local Whisper (faster-whisper).

    Model is loaded lazily on first use and unloaded after idle timeout
    (WHISPER_IDLE_TIMEOUT) to free memory. Runs on CPU to avoid GPU contention
    with the RAG embedding/reranker models.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "STTService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._model = None
                    cls._instance._last_used = 0.0
                    cls._instance._unloader = None
        return cls._instance

    # ── Model loading ──

    def _load_model(self):
        """Load the Whisper model into memory."""
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ValueError("未安装 faster-whisper，请先 pip install faster-whisper") from e

        model_name = settings.WHISPER_MODEL_NAME
        # 自动选设备：有 GPU 用 float16，否则 CPU int8
        device = os.getenv("WHISPER_DEVICE", "auto")
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        compute_type = "int8" if device == "cpu" else "float16"
        logger.info("加载本地 Whisper 模型: %s (device=%s, compute=%s)",
                    model_name, device, compute_type)
        return WhisperModel(model_name, device=device, compute_type=compute_type)

    def _ensure_model(self):
        with self._lock:
            if self._model is None:
                self._model = self._load_model()
            self._last_used = time.time()
            return self._model

    def _schedule_unload(self):
        """Schedule model unload after idle timeout (single daemon thread)."""
        with self._lock:
            if self._unloader is not None and self._unloader.is_alive():
                return
            self._unloader = threading.Thread(target=self._idle_unload, daemon=True)
            self._unloader.start()

    def _idle_unload(self):
        time.sleep(settings.WHISPER_IDLE_TIMEOUT)
        with self._lock:
            if self._model is not None and time.time() - self._last_used >= settings.WHISPER_IDLE_TIMEOUT:
                logger.info("Whisper 空闲超时，卸载模型释放内存")
                self._model = None
                gc.collect()
            self._unloader = None

    # ── Transcription ──

    async def transcribe(self, file: UploadFile) -> str:
        """Transcribe an uploaded audio file to text."""
        content = await file.read()
        suffix = ".webm"
        if file.filename and "." in file.filename:
            suffix = "." + file.filename.rsplit(".", 1)[-1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            return await asyncio.to_thread(self._transcribe_path, tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    def _transcribe_path(self, path: str) -> str:
        model = self._ensure_model()
        try:
            segments, _info = model.transcribe(
                path, language=settings.WHISPER_LANGUAGE, beam_size=5
            )
            text = "".join(seg.text for seg in segments).strip()
            return self._to_simplified(text)
        finally:
            self._schedule_unload()

    @staticmethod
    def _to_simplified(text: str) -> str:
        """繁体 → 简体（Whisper 偶发输出繁体字）。"""
        if not text:
            return text
        try:
            from opencc import OpenCC
            return OpenCC("t2s").convert(text)
        except Exception:
            return text
