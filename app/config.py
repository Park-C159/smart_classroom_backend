"""Application configuration loaded from environment variables."""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──
    APP_NAME: str = "数学教材智能答疑助手"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    SECRET_KEY: str = "change-me-in-production"
    ENVIRONMENT: str = "development"  # development | production | testing

    # ── Database ──
    DATABASE_URL: str = "sqlite+aiosqlite:///./math_qa.db"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40

    # ── Redis ──
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── JWT ──
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # ── DeepSeek API ──
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_CHAT_MODEL: str = "deepseek-v4-pro"
    DEEPSEEK_REASONER_MODEL: str = "deepseek-v4-pro"

    # ── 联网搜索（百度 AI 搜索 API）──
    BAIDU_SEARCH_API_KEY: str = ""   # 百度智能云 API Key；为空时回退 Bing 抓取

    # ── VLM（图片识别）── 为空时回退到 DEEPSEEK_* 配置
    VLM_MODEL: str = ""             # 视觉模型名（需支持图片输入，如 qwen-vl-max）
    VLM_BASE_URL: str = ""          # 视觉模型 base_url

    # ── GPU / Models ──
    CUDA_DEVICE: str = "cuda:0"
    BGE_M3_MODEL_NAME: str = "BAAI/bge-m3"
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-v2-m3"
    WHISPER_MODEL_NAME: str = "medium"  # tiny | base | small | medium | large-v3
    WHISPER_LANGUAGE: str = "zh"       # 识别语言：zh = 简体中文
    WHISPER_IDLE_TIMEOUT: int = 300    # seconds before unloading to CPU

    # ── FAISS ──
    VECTOR_STORE_DIR: str = str(Path(__file__).parent.parent / "data" / "vector_store")

    # ── MinerU ──
    MINERU_MODE: str = "high-quality"  # high-quality | fast
    MINERU_API_URL: str = ""           # 常驻 mineru-api 服务地址（如 http://127.0.0.1:8002）；空则每次起临时服务
    MINERU_BIN: str = ""               # mineru CLI 路径（MinerU 独立环境）；空则自动检测
    MINERU_API_BIN: str = ""           # mineru-api 路径（MinerU 独立环境）；空则自动检测

    # ── RAG ──
    RAG_TOP_K_KP: int = 3              # stage-1: top-K knowledge points
    RAG_KP_THRESHOLD: float = 0.6      # min similarity for KP match
    RAG_CONTENT_CANDIDATES: int = 30   # stage-2: content recall pool
    RAG_EXERCISE_CANDIDATES: int = 20  # stage-2: exercise recall pool
    RAG_MAX_CANDIDATES: int = 8        # max candidates entering reranker
    RAG_RERANK_TOP_K: int = 5          # final top-K after reranker
    CHAT_HISTORY_ROUNDS: int = 3       # conversation rounds injected into prompt

    # ── Upload ──
    MAX_UPLOAD_SIZE_MB: int = 500      # PDF max size
    ALLOWED_UPLOAD_EXTENSIONS: list[str] = ["pdf", "xlsx", "xls", "csv"]
    DATA_DIR: str = str(Path(__file__).parent.parent / "data")

    # ── Celery ──
    CELERY_BROKER_URL: str = "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/2"

    # ── CORS ──
    CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:5175", "http://localhost:8000", "http://localhost:3000"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "allow"}


settings = Settings()
