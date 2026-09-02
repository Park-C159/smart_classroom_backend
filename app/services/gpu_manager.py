"""GPU memory manager — moves models between CPU/GPU based on task needs.

Single RTX 5070 Laptop 12GB VRAM constraint: models are loaded to GPU only when
needed and moved back to CPU when not in use. Thread-safe via asyncio.Lock.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections import OrderedDict
from typing import Any

import torch

logger = logging.getLogger(__name__)


class GPUManager:
    """Singleton GPU memory scheduler.

    Usage:
        gpu = GPUManager()
        gpu.register("embedding", bge_model)
        gpu.to_gpu("embedding")          # move specific model to GPU
        gpu.clear_gpu()                  # all models -> CPU (before MinerU)
        gpu.restore_defaults()           # restore resident models -> GPU
    """

    _instance: GPUManager | None = None
    _lock = threading.Lock()

    def __new__(cls) -> GPUManager:
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._models: OrderedDict[str, Any] = OrderedDict()
        self._defaults: list[str] = []  # models to keep on GPU by default
        self._task_lock = asyncio.Lock()
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Track which models are currently on GPU
        self._on_gpu: set[str] = set()

    # ── Registration ──

    def register(self, name: str, model: Any, *, default_on_gpu: bool = False) -> None:
        """Register a model. If default_on_gpu=True, it stays on GPU in daily service."""
        self._models[name] = model
        if default_on_gpu:
            self._defaults.append(name)
            self._on_gpu.add(name)
        logger.info(f"GPUManager: registered '{name}' (default_on_gpu={default_on_gpu})")

    def unregister(self, name: str) -> None:
        """Remove a model from tracking."""
        self._models.pop(name, None)
        self._on_gpu.discard(name)
        if name in self._defaults:
            self._defaults.remove(name)

    # ── Device movement ──

    def to_cpu(self, name: str) -> None:
        """Move a single model to CPU."""
        if name not in self._models:
            return
        model = self._models[name]
        if hasattr(model, "to"):
            model.to("cpu")
        self._on_gpu.discard(name)
        logger.debug(f"GPUManager: '{name}' → CPU")

    def to_gpu(self, name: str) -> None:
        """Move a single model to GPU."""
        if name not in self._models:
            return
        model = self._models[name]
        if hasattr(model, "to"):
            model.to(self._device)
        self._on_gpu.add(name)
        logger.debug(f"GPUManager: '{name}' → GPU")

    # ── Batch operations ──

    def clear_gpu(self) -> None:
        """Move ALL models to CPU and release cached VRAM. Call before MinerU or FAISS rebuild."""
        for name in list(self._models.keys()):
            self.to_cpu(name)
        # 释放 PyTorch 缓存分配器占用的显存，让 MinerU（子进程/同进程）能拿到
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("GPUManager: all models moved to CPU")

    def restore_defaults(self) -> None:
        """Restore default resident models to GPU."""
        for name in self._defaults:
            if name in self._models:
                self.to_gpu(name)
        logger.info(f"GPUManager: restored defaults to GPU: {self._defaults}")

    # ── Async context manager for task isolation ──

    async def acquire_task(self, required: list[str], *, free_others: bool = True) -> None:
        """Acquire GPU for a task. Moves required models to GPU, optionally frees others.

        Args:
            required: list of model names needed on GPU
            free_others: if True, move all other models to CPU first
        """
        await self._task_lock.acquire()
        try:
            if free_others:
                # Move everything else to CPU
                for name in list(self._on_gpu):
                    if name not in required:
                        self.to_cpu(name)
            # Move required models to GPU
            for name in required:
                self.to_gpu(name)
        except Exception:
            self._task_lock.release()
            raise

    def release_task(self) -> None:
        """Release GPU task lock and restore default state."""
        try:
            self.restore_defaults()
        finally:
            if self._task_lock.locked():
                self._task_lock.release()

    # ── Status ──

    def gpu_memory_used(self) -> float:
        """Return GPU memory used in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.memory_allocated() / (1024 ** 3)

    def gpu_memory_total(self) -> float:
        """Return total GPU memory in GB."""
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    @property
    def on_gpu(self) -> set[str]:
        return self._on_gpu.copy()

    @property
    def device(self) -> torch.device:
        return self._device


# Global singleton
gpu_manager = GPUManager()
