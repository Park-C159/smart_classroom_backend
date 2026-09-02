"""Document processing Celery tasks — MinerU parsing + index rebuild."""
from app.tasks.schedule import celery_app


@celery_app.task(name="app.tasks.document_tasks.parse_pending_pdfs")
def parse_pending_pdfs():
    """Nightly task: parse all PDFs with status='pending' using MinerU.

    Steps:
    1. GPUManager.clear_gpu() — free all GPU memory
    2. For each pending PDF:
       a. MinerU(mode='high-quality') → Markdown + LaTeX
       b. DeepSeek API validation — check for LaTeX errors
       c. Save results to data/parsed/
       d. Update document status
    3. Trigger FAISS index rebuild
    4. GPUManager.restore_defaults()
    """
    # Will be implemented in Stage 2
    pass


@celery_app.task(name="app.tasks.document_tasks.rebuild_faiss_indexes")
def rebuild_faiss_indexes(document_ids: list[int] | None = None):
    """Rebuild all 3 FAISS indexes after new documents are added.

    Args:
        document_ids: if provided, only rebuild chunks from these docs (then merge)
    """
    # Will be implemented in Stage 2
    pass


@celery_app.task(name="app.tasks.document_tasks.build_knowledge_tree")
def build_knowledge_tree(document_id: int):
    """Build knowledge tree from parsed document:
    1. Rule engine: extract H1/H2/H3 → tree skeleton
    2. Regex: tag blocks as definition/theorem/example
    3. DeepSeek API: generate KP summaries, match exercises → KPs
    4. Save to PostgreSQL
    5. Trigger FAISS rebuild
    """
    # Will be implemented in Stage 2
    pass
