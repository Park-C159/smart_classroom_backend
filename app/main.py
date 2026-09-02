"""FastAPI application entry point."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.api import auth, users, rag, document, subjects, speech, upload, discussion, analytics, exam, knowledge, feedback
from app.api import test_bank, papers, messages


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup/shutdown events."""
    # Startup
    from app.core.database import init_db
    await init_db()

    # Pre-load RAG models to CPU, auto-move to GPU on first use
    print("Loading RAG models...")
    from app.services.rag_service import RAGService
    from app.services.gpu_manager import gpu_manager
    rag = RAGService()
    gpu_manager.register("embedding", rag.embedding_model, default_on_gpu=True)
    gpu_manager.register("reranker", rag.reranker, default_on_gpu=True)
    # Move to GPU now so first query is fast
    import torch
    if torch.cuda.is_available():
        rag.embedding_model.to("cuda")
        rag.reranker.to("cuda")
        print(f"RAG models on GPU, VRAM: {gpu_manager.gpu_memory_used():.1f}G")
    else:
        print("RAG models on CPU")

    # 预加载本地 Whisper 模型（后台线程，避免阻塞启动；语音识别首用更快）
    import threading
    def _preload_stt():
        try:
            from app.services.stt_service import STTService
            STTService()._ensure_model()
            print("Whisper 模型预加载完成")
        except Exception as e:
            print(f"Whisper 预加载失败（首次使用时再加载）: {e}")
    threading.Thread(target=_preload_stt, daemon=True).start()

    yield
    # Shutdown
    from app.core.redis_client import redis_close
    await redis_close()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(rag.router)
app.include_router(document.router)
app.include_router(subjects.router)
app.include_router(speech.router)
app.include_router(upload.router)
app.include_router(discussion.router)
app.include_router(analytics.router)
app.include_router(exam.router)
app.include_router(knowledge.router)
app.include_router(feedback.router)
app.include_router(test_bank.router)
app.include_router(papers.router)
app.include_router(messages.router)


@app.get("/api/health")
async def health_check():
    """Health check endpoint for load balancers/monitoring."""
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@app.get("/")
async def root():
    """Root redirect to API docs."""
    return {
        "message": f"欢迎使用{settings.APP_NAME} API",
        "docs": "/docs",
        "version": settings.APP_VERSION,
    }
