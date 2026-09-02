"""Speech recognition API."""
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.core.security import get_current_user

router = APIRouter(prefix="/api/speech", tags=["语音识别"])


@router.post("/recognize")
async def recognize_speech(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Transcribe audio to text. Alias for /api/chat/transcribe."""
    from app.api.rag import transcribe_audio
    return await transcribe_audio(file=file, current_user=current_user)
