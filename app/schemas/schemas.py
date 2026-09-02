"""Pydantic schemas for request/response validation."""
from datetime import datetime
from pydantic import BaseModel, Field


# ── Auth ──

class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., min_length=1)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: "UserOut"


class RefreshRequest(BaseModel):
    refresh_token: str


# ── User ──

class UserCreate(BaseModel):
    username: str = Field(..., min_length=2, max_length=50)
    password: str = Field(..., min_length=6)
    real_name: str = Field(default="", max_length=50)
    role: str = Field(default="student", pattern="^(student|teacher|admin)$")
    class_name: str | None = None
    student_id: str | None = None
    email: str | None = None


class UserUpdate(BaseModel):
    real_name: str | None = Field(None, max_length=50)
    class_name: str | None = None
    student_id: str | None = None
    email: str | None = None
    is_active: bool | None = None
    role: str | None = Field(None, pattern="^(student|teacher|admin)$")


class UserOut(BaseModel):
    id: int
    username: str
    real_name: str
    role: str
    class_name: str | None = None
    student_id: str | None = None
    email: str | None = None
    avatar_url: str | None = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class UserBatchImportRow(BaseModel):
    username: str
    real_name: str
    role: str = "student"
    class_name: str | None = None
    student_id: str | None = None
    password: str = "123456"


class ImportResult(BaseModel):
    total: int
    success: int
    failed: int
    errors: list[str] = []


# ── Knowledge Tree ──

class KPNodeBase(BaseModel):
    id: str
    title: str
    summary: str | None = None
    parent_id: str | None = None
    chapter: str | None = None
    level: int = 0
    sort_order: int = 0


class KPNodeCreate(KPNodeBase):
    pass


class KPNodeOut(KPNodeBase):
    created_at: datetime

    model_config = {"from_attributes": True}


class KnowledgeTreeOut(BaseModel):
    nodes: list[KPNodeOut]
    edges: list[dict]  # [{source: KP-1.1, target: KP-1.2}]


# ── Content Chunk ──

class ContentChunkOut(BaseModel):
    id: int
    kp_id: str
    chunk_type: str
    content: str
    page_number: int | None = None
    faiss_id: int | None = None

    model_config = {"from_attributes": True}


# ── Exercise ──

class ExerciseCreate(BaseModel):
    kp_id: str
    question_text: str
    answer_text: str | None = None
    question_type: str = "calculation"
    difficulty: int = Field(default=3, ge=1, le=5)
    source: str = "teacher"
    page_number: int | None = None


class ExerciseOut(BaseModel):
    id: int
    kp_id: str
    question_text: str
    answer_text: str | None = None
    question_type: str
    difficulty: int
    source: str
    page_number: int | None = None
    faiss_id: int | None = None

    model_config = {"from_attributes": True}


# ── RAG Q&A ──

class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    subject_id: int | None = None
    document_ids: list[int] = []
    use_deep_think: bool = False
    use_smart_search: bool = True
    conversation_id: int | None = None


class Citation(BaseModel):
    chunk_id: int
    content: str
    kp_id: str
    kp_title: str
    page_number: int | None = None


class QAResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    matched_kps: list[str] = []  # ['KP-2.3', 'KP-2.2']
    recommended_exercises: list[ExerciseOut] = []


# ── Discussion ──

class DiscussionCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    subject_id: int | None = None
    kp_id: str | None = None
    qa_refs: list[dict] | None = None


class DiscussionOut(BaseModel):
    id: int
    title: str
    content: str
    user_id: int
    author_name: str = ""
    author_avatar: str | None = None
    subject_id: int | None = None
    kp_id: str | None = None
    qa_refs: list[dict] | None = None
    is_pinned: bool
    view_count: int
    like_count: int
    reply_count: int
    is_liked: bool = False
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class DiscussionReplyCreate(BaseModel):
    content: str = Field(..., min_length=1)
    parent_id: int | None = None


class DiscussionReplyOut(BaseModel):
    id: int
    discussion_id: int
    user_id: int
    author_name: str = ""
    author_avatar: str | None = None
    content: str
    parent_id: int | None = None
    like_count: int
    is_liked: bool = False
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Feedback ──

class FeedbackCreate(BaseModel):
    category: str = Field(default="feature", pattern="^(feature|bug|experience)$")
    content: str = Field(..., min_length=1, max_length=2000)
    attachment: str | None = None


class FeedbackOut(BaseModel):
    id: int
    user_id: int
    category: str
    content: str
    attachment: str | None = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Parsing / Validation ──

class ParseError(BaseModel):
    line: int
    type: str  # latex_concat | bracket_mismatch | chapter_skip | paragraph_merge | header_footer
    suggestion: str
    original: str = ""


class ValidationReport(BaseModel):
    document_id: int
    errors: list[ParseError] = []
    status: str = "ok"  # ok | has_errors | fixed


# ── Pagination ──

class PaginatedResponse(BaseModel):
    items: list
    total: int
    page: int
    page_size: int
    total_pages: int
