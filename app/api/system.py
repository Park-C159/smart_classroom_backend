"""System monitoring & functional logs API (admin only)."""
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, Query

from app.config import settings
from app.core.security import get_admin_user
from app.services.gpu_manager import gpu_manager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/system", tags=["系统监测"])


def _tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.readlines()
        return "".join(content[-lines:])
    except Exception as e:
        return f"读取日志失败: {e}"


@router.get("/status")
async def system_status(current_user: dict = Depends(get_admin_user)):
    """Return system health: GPU / CPU / memory / app info."""
    import platform

    gpu = {
        "available": False,
        "device": "cpu",
        "memory_used_gb": 0.0,
        "memory_total_gb": 0.0,
        "models_on_gpu": [],
    }
    try:
        import torch
        gpu["available"] = torch.cuda.is_available()
        gpu["device"] = gpu_manager.device.type
        gpu["memory_used_gb"] = round(gpu_manager.gpu_memory_used(), 2)
        gpu["memory_total_gb"] = round(gpu_manager.gpu_memory_total(), 2)
        gpu["models_on_gpu"] = sorted(gpu_manager.on_gpu)
    except Exception as e:
        gpu["error"] = str(e)

    cpu_mem = {}
    try:
        import psutil
        vm = psutil.virtual_memory()
        cpu_mem = {
            "cpu_percent": psutil.cpu_percent(interval=0.2),
            "memory_percent": vm.percent,
            "memory_used_gb": round(vm.used / (1024 ** 3), 2),
            "memory_total_gb": round(vm.total / (1024 ** 3), 2),
        }
    except ImportError:
        cpu_mem = {"note": "psutil 未安装，无法读取 CPU/内存"}
    except Exception as e:
        cpu_mem = {"error": str(e)}

    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "python": platform.python_version(),
        "os": platform.platform(),
        "gpu": gpu,
        "cpu_memory": cpu_mem,
    }


@router.get("/logs")
async def system_logs(
    lines: int = Query(200, ge=1, le=5000),
    current_user: dict = Depends(get_admin_user),
):
    """Tail the backend functional log file."""
    log_path = Path(settings.DATA_DIR).parent / "backend.log"
    content = _tail_file(log_path, lines)
    return {
        "path": str(log_path),
        "exists": log_path.exists(),
        "line_count": len(content.splitlines()),
        "log": content,
    }
