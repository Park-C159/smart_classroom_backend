"""SQLAlchemy models for all database tables."""
from datetime import datetime, timezone

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# ── User & Auth ──

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    real_name: Mapped[str] = mapped_column(String(50), default="")
    role: Mapped[str] = mapped_column(String(20), default="student", index=True)  # student | teacher | admin
    class_name: Mapped[str | None] = mapped_column(String(100))
    student_id: Mapped[str | None] = mapped_column(String(50))
    email: Mapped[str | None] = mapped_column(String(100))
    avatar_url: Mapped[str | None] = mapped_column(String(500))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # relationships
    interactions: Mapped[list["InteractionLog"]] = relationship(back_populates="user")
    kp_masteries: Mapped[list["KPMastery"]] = relationship(back_populates="user")
    discussions: Mapped[list["Discussion"]] = relationship(back_populates="author", foreign_keys="Discussion.user_id")


# ── Knowledge Tree ──

class KnowledgePoint(Base):
    __tablename__ = "knowledge_points"

    id: Mapped[str] = mapped_column(String(20), primary_key=True)  # e.g. 'KP-2.3'
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[str | None] = mapped_column(String(20), ForeignKey("knowledge_points.id"), index=True)
    chapter: Mapped[str | None] = mapped_column(String(50))
    level: Mapped[int] = mapped_column(Integer, default=0)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # relationships
    children: Mapped[list["KnowledgePoint"]] = relationship(
        "KnowledgePoint", back_populates="parent", remote_side=[id], foreign_keys=[parent_id]
    )
    parent: Mapped["KnowledgePoint | None"] = relationship(
        "KnowledgePoint", back_populates="children", remote_side=[parent_id], foreign_keys=[parent_id]
    )
    content_chunks: Mapped[list["ContentChunk"]] = relationship(back_populates="knowledge_point")
    exercises: Mapped[list["Exercise"]] = relationship(back_populates="knowledge_point")
    masteries: Mapped[list["KPMastery"]] = relationship(back_populates="knowledge_point")


class ContentChunk(Base):
    __tablename__ = "content_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kp_id: Mapped[str] = mapped_column(String(20), ForeignKey("knowledge_points.id"), nullable=False, index=True)
    chunk_type: Mapped[str] = mapped_column(String(20), default="definition")  # definition|theorem|example|remark
    content: Mapped[str] = mapped_column(Text, nullable=False)
    page_number: Mapped[int | None] = mapped_column(Integer)
    faiss_id: Mapped[int | None] = mapped_column(Integer)
    faiss_kb_id: Mapped[int | None] = mapped_column(Integer)          # KB vector index (separate from QB)
    subject_id: Mapped[int | None] = mapped_column(Integer, index=True)  # Subject for KB index scoping
    source_doc_id: Mapped[int | None] = mapped_column(Integer)  # source document ID (1=primary, 2=reference)
    images: Mapped[dict | None] = mapped_column(JSON)  # [{"path": "...", "vlm_desc": "..."}]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    knowledge_point: Mapped["KnowledgePoint"] = relationship(back_populates="content_chunks")


class Exercise(Base):
    __tablename__ = "exercises"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    kp_id: Mapped[str] = mapped_column(String(20), ForeignKey("knowledge_points.id"), nullable=False, index=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(20), default="calculation")  # choice|fill|proof|calculation
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    source: Mapped[str] = mapped_column(String(20), default="textbook")  # textbook | teacher
    page_number: Mapped[int | None] = mapped_column(Integer)
    faiss_id: Mapped[int | None] = mapped_column(Integer)
    images: Mapped[dict | None] = mapped_column(JSON)  # [{"path": "...", "vlm_desc": "..."}]
    embedding_text: Mapped[str | None] = mapped_column(Text)  # text for FAISS (includes VLM desc)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    source_doc_id: Mapped[int | None] = mapped_column(Integer)  # source document ID
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    knowledge_point: Mapped["KnowledgePoint"] = relationship(back_populates="exercises")


# ── Analytics ──

class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str | None] = mapped_column(Text)
    matched_kps: Mapped[dict | None] = mapped_column(JSON)
    cited_chunks: Mapped[dict | None] = mapped_column(JSON)
    feedback: Mapped[str | None] = mapped_column(String(20))  # helpful | not_helpful | null
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(back_populates="interactions")


class KPMastery(Base):
    __tablename__ = "kp_mastery"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    kp_id: Mapped[str] = mapped_column(String(20), ForeignKey("knowledge_points.id"), primary_key=True)
    mastery: Mapped[float] = mapped_column(Float, default=0.5)
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    user: Mapped["User"] = relationship(back_populates="kp_masteries")
    knowledge_point: Mapped["KnowledgePoint"] = relationship(back_populates="masteries")


# ── Discussion ──

class Discussion(Base):
    __tablename__ = "discussions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    subject_id: Mapped[int | None] = mapped_column(Integer)
    kp_id: Mapped[str | None] = mapped_column(String(20))
    qa_refs: Mapped[dict | None] = mapped_column(JSON)
    is_pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    view_count: Mapped[int] = mapped_column(Integer, default=0)
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    reply_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    author: Mapped["User"] = relationship(back_populates="discussions", foreign_keys=[user_id])
    replies: Mapped[list["DiscussionReply"]] = relationship(back_populates="discussion")
    likes: Mapped[list["DiscussionLike"]] = relationship(back_populates="discussion")


class DiscussionReply(Base):
    __tablename__ = "discussion_replies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    discussion_id: Mapped[int] = mapped_column(Integer, ForeignKey("discussions.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    parent_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("discussion_replies.id"))
    like_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    discussion: Mapped["Discussion"] = relationship(back_populates="replies")
    user: Mapped["User"] = relationship()
    parent: Mapped["DiscussionReply | None"] = relationship("DiscussionReply", remote_side=[id])


class DiscussionLike(Base):
    __tablename__ = "discussion_likes"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    discussion_id: Mapped[int] = mapped_column(Integer, ForeignKey("discussions.id", ondelete="CASCADE"), primary_key=True)

    discussion: Mapped["Discussion"] = relationship(back_populates="likes")


class ReplyLike(Base):
    __tablename__ = "reply_likes"

    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), primary_key=True)
    reply_id: Mapped[int] = mapped_column(Integer, ForeignKey("discussion_replies.id", ondelete="CASCADE"), primary_key=True)


# ── Subject ──

class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    primary_doc_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("documents.id"), default=None)  # Primary textbook
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── Document ──

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), default="")
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    upload_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    doc_type: Mapped[str | None] = mapped_column(String(20), default=None, index=True)  # textbook | reference | None
    subject_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("subjects.id"), default=None, index=True)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=False)  # Primary textbook for subject
    status: Mapped[str] = mapped_column(String(20), default="pending", index=True)
    total_pages: Mapped[int] = mapped_column(Integer, default=0)
    processed_pages: Mapped[int] = mapped_column(Integer, default=0)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    uploader: Mapped["User"] = relationship(foreign_keys=[upload_by])


# ── Association tables ──

class DocumentSubject(Base):
    __tablename__ = "document_subjects"
    document_id: Mapped[int] = mapped_column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    subject_id: Mapped[int] = mapped_column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True)


class UserSubject(Base):
    __tablename__ = "user_subjects"
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    subject_id: Mapped[int] = mapped_column(Integer, ForeignKey("subjects.id", ondelete="CASCADE"), primary_key=True)


# ── Question Bank ──

class QuestionBank(Base):
    """Deduplicated question bank — questions separated from answers, linked to chapters."""
    __tablename__ = "question_bank"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text)           # Separated answer (None if no answer)
    question_type: Mapped[str] = mapped_column(String(20), default="calculation")  # calculation|proof|choice|fill
    difficulty: Mapped[int] = mapped_column(Integer, default=3)     # 1-5
    source: Mapped[str] = mapped_column(String(20), default="textbook")  # textbook | reference | teacher
    source_doc_id: Mapped[int | None] = mapped_column(Integer)       # Source document
    page_number: Mapped[int | None] = mapped_column(Integer)         # PDF page
    chapter: Mapped[str | None] = mapped_column(String(100), index=True)  # Chapter only (e.g. "第一章 多项式")
    kp_id: Mapped[str | None] = mapped_column(String(20), ForeignKey("knowledge_points.id"), index=True)
    subject_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("subjects.id"), index=True)
    images: Mapped[dict | None] = mapped_column(JSON)               # [{"path": "...", "vlm_desc": "..."}]
    faiss_id: Mapped[int | None] = mapped_column(Integer)            # QB vector index ID
    embedding_text: Mapped[str | None] = mapped_column(Text)         # Text used for embedding (incl. VLM desc)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    merged_from: Mapped[list | None] = mapped_column(JSON)           # [source question IDs merged]
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


# ── Exam ──

class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    subject_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("subjects.id"))
    kp_ids: Mapped[dict | None] = mapped_column(JSON)  # list of KP IDs covered
    total_questions: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int | None] = mapped_column(Integer)
    score: Mapped[float | None] = mapped_column(Float)  # 0-100
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)  # draft | submitted | graded
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    # passive_deletes: SQLite 不启用外键级联，删 exam 时别让 SQLAlchemy 去置空 children 的 FK
    questions: Mapped[list["ExamQuestion"]] = relationship(back_populates="exam", passive_deletes=True)


class ExamQuestion(Base):
    __tablename__ = "exam_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    exam_id: Mapped[int] = mapped_column(Integer, ForeignKey("exams.id", ondelete="CASCADE"), nullable=False, index=True)
    exercise_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("exercises.id"))
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    answer_text: Mapped[str | None] = mapped_column(Text)
    question_type: Mapped[str] = mapped_column(String(20), default="calculation")
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    sort_order: Mapped[int] = mapped_column(Integer, default=0)
    user_answer: Mapped[str | None] = mapped_column(Text)
    is_correct: Mapped[bool | None] = mapped_column(Boolean)

    exam: Mapped["Exam"] = relationship(back_populates="questions")
    exercise: Mapped["Exercise | None"] = relationship(foreign_keys=[exercise_id])


# ── Feedback ──

class Feedback(Base):
    __tablename__ = "feedbacks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(20), default="feature")  # feature|bug|experience
    content: Mapped[str] = mapped_column(Text, nullable=False)
    attachment: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    user: Mapped["User"] = relationship(foreign_keys=[user_id])


# ── 独立试题库（教师/管理员上传，用于组卷/作业/测试/考试）──

class TestQuestion(Base):
    __tablename__ = "test_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("subjects.id"), index=True)
    chapter: Mapped[str | None] = mapped_column(String(100), index=True)
    kp_id: Mapped[str | None] = mapped_column(String(20), ForeignKey("knowledge_points.id"), index=True)
    question_type: Mapped[str] = mapped_column(String(20), default="choice", index=True)  # choice | fill | short_answer
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list | None] = mapped_column(JSON)  # choice: [{"key":"A","text":"..."}, ...]
    answer_text: Mapped[str] = mapped_column(Text, nullable=False)  # choice=正确key 如"A"/"AC"；fill=期望答案；short_answer=参考答案
    difficulty: Mapped[int] = mapped_column(Integer, default=3)  # 1-5
    images: Mapped[dict | None] = mapped_column(JSON)
    created_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"))
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


# ── 组卷（作业/测试/考试/练习）──

class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    subject_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("subjects.id"))
    mode: Mapped[str] = mapped_column(String(20), default="practice", index=True)  # practice | homework | test | exam
    chapter: Mapped[str | None] = mapped_column(String(100))          # 章节筛选
    difficulty_min: Mapped[int] = mapped_column(Integer, default=1)
    difficulty_max: Mapped[int] = mapped_column(Integer, default=5)
    created_by: Mapped[int | None] = mapped_column(Integer, ForeignKey("users.id"), index=True)  # 练习=null，作业/测试/考试=教师
    target_class: Mapped[str | None] = mapped_column(String(100))     # 作业/测试/考试指定班级；练习为空
    published: Mapped[bool] = mapped_column(Boolean, default=False)
    due_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    questions: Mapped[list["PaperQuestion"]] = relationship(back_populates="paper", passive_deletes=True)
    submissions: Mapped[list["PaperSubmission"]] = relationship(back_populates="paper", passive_deletes=True)


class PaperQuestion(Base):
    __tablename__ = "paper_questions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    test_question_id: Mapped[int | None] = mapped_column(Integer, ForeignKey("test_questions.id"))
    question_text: Mapped[str] = mapped_column(Text, nullable=False)
    question_type: Mapped[str] = mapped_column(String(20), default="choice")
    options: Mapped[list | None] = mapped_column(JSON)
    answer_text: Mapped[str | None] = mapped_column(Text)
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    kp_id: Mapped[str | None] = mapped_column(String(20))  # 从 TestQuestion 复制的知识点（用于逐题掌握度）
    sort_order: Mapped[int] = mapped_column(Integer, default=0)

    paper: Mapped["Paper"] = relationship(back_populates="questions")


class PaperSubmission(Base):
    __tablename__ = "paper_submissions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    paper_id: Mapped[int] = mapped_column(Integer, ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), default="draft", index=True)  # draft | submitted
    score: Mapped[float | None] = mapped_column(Float)  # 0-100
    correct_count: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    paper: Mapped["Paper"] = relationship(back_populates="submissions")
    answers: Mapped[list["PaperAnswer"]] = relationship(back_populates="submission", passive_deletes=True)


class PaperAnswer(Base):
    __tablename__ = "paper_answers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    submission_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_submissions.id", ondelete="CASCADE"), nullable=False, index=True)
    paper_question_id: Mapped[int] = mapped_column(Integer, ForeignKey("paper_questions.id"), nullable=False)
    user_answer: Mapped[str | None] = mapped_column(Text)
    is_correct: Mapped[bool | None] = mapped_column(Boolean)
    score: Mapped[float | None] = mapped_column(Float)  # 客观 0/1，简答 LLM 0-1
    feedback: Mapped[str | None] = mapped_column(Text)
    graded_by: Mapped[str | None] = mapped_column(String(20))  # auto | llm | teacher
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    submission: Mapped["PaperSubmission"] = relationship(back_populates="answers")


# ── 私信 ──

class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    sender_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    recipient_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_read: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sender: Mapped["User"] = relationship(foreign_keys=[sender_id])
    recipient: Mapped["User"] = relationship(foreign_keys=[recipient_id])
