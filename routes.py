# api/routes.py
"""
Step 11: FastAPI Route Handlers
All API endpoints with authentication, validation, and proper error handling.
"""

import os
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from loguru import logger

from api.auth import get_current_user, get_optional_user, require_admin, TokenData, authenticate_user, create_access_token
from api.models import (
    LoginRequest, LoginResponse,
    QueryRequest, QueryResponse,
    IngestResponse, IngestDirectoryRequest,
    FeedbackRequest, FeedbackResponse,
    HealthResponse, SourcesResponse,
)
from core.config import settings
from core.document_processor import DocumentProcessor
from core.memory import get_memory_manager

router = APIRouter()

# ──────────────────────────────────────────────
# Lazy-loaded orchestrator (singleton)
# ──────────────────────────────────────────────

_orchestrator = None

def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        from agents.orchestrator import MultiAgentOrchestrator
        _orchestrator = MultiAgentOrchestrator()
        logger.info("MultiAgentOrchestrator initialized")
    return _orchestrator


# ──────────────────────────────────────────────
# Auth Endpoints
# ──────────────────────────────────────────────

@router.post("/auth/login", response_model=LoginResponse, tags=["Auth"])
async def login(credentials: LoginRequest):
    """
    Authenticate and receive a JWT token.
    Use the token in the Authorization: Bearer <token> header.
    """
    user = authenticate_user(credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )

    token = create_access_token({
        "user_id": user["user_id"],
        "username": credentials.username,
        "roles": user["roles"],
        "department": user.get("department"),
    })

    return LoginResponse(
        access_token=token,
        expires_in=settings.access_token_expire_minutes * 60,
        user_id=user["user_id"],
        username=credentials.username,
        roles=user["roles"],
    )


# ──────────────────────────────────────────────
# Query Endpoint
# ──────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(
    request: QueryRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """
    Submit a natural language question to the knowledge assistant.
    
    The pipeline:
    1. RetrievalAgent fetches relevant chunks (hybrid search)
    2. ValidationAgent reranks and filters
    3. ResponseAgent generates cited answer
    
    Returns answer with source citations.
    """
    memory = get_memory_manager()

    # Load conversation history
    history = []
    if request.use_history:
        history = await memory.get_history(request.session_id, last_n=10)

    # Run multi-agent pipeline
    try:
        orchestrator = get_orchestrator()
        result = orchestrator.run(
            query=request.query,
            user_roles=current_user.roles,
            conversation_history=history,
            session_id=request.session_id,
        )
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Persist messages to memory
    await memory.save_message(
        session_id=request.session_id,
        role="user",
        content=request.query,
        user_id=current_user.user_id,
    )
    await memory.save_message(
        session_id=request.session_id,
        role="assistant",
        content=result.get("answer", ""),
        user_id=current_user.user_id,
        sources=result.get("sources", []),
        confidence=result.get("confidence"),
    )

    return QueryResponse(**result)


# ──────────────────────────────────────────────
# Document Ingestion Endpoints
# ──────────────────────────────────────────────

@router.post("/ingest/file", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    department: str = Form(default=None),
    access_roles: str = Form(default="all"),   # comma-separated
    current_user: TokenData = Depends(require_admin),
):
    """
    Upload and ingest a single document (PDF, DOCX, TXT, MD).
    Admin role required.
    """
    allowed_extensions = {".pdf", ".docx", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}",
        )

    # Save uploaded file temporarily
    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path = upload_dir / f"{uuid.uuid4()}{file_ext}"

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        # Process & ingest
        roles = [r.strip() for r in access_roles.split(",") if r.strip()]
        processor = DocumentProcessor()
        chunks = processor.process_file(
            str(temp_path),
            department=department,
            access_roles=roles,
        )

        if not chunks:
            raise HTTPException(status_code=422, detail="No text could be extracted")

        # Override source name to original filename
        for chunk in chunks:
            chunk.source = file.filename

        orchestrator = get_orchestrator()
        orchestrator.add_documents(chunks)

        return IngestResponse(
            message=f"Successfully ingested {file.filename}",
            chunks_created=len(chunks),
            source=file.filename,
            department=department,
        )

    finally:
        if temp_path.exists():
            os.remove(temp_path)


@router.post("/ingest/directory", response_model=IngestResponse, tags=["Ingestion"])
async def ingest_directory(
    request: IngestDirectoryRequest,
    current_user: TokenData = Depends(require_admin),
):
    """
    Recursively ingest all documents in a server-side directory.
    Admin role required.
    """
    if not Path(request.directory).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Directory not found: {request.directory}",
        )

    processor = DocumentProcessor()
    chunks = processor.process_directory(
        request.directory,
        department=request.department,
        access_roles=request.access_roles,
    )

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No documents found or no text could be extracted",
        )

    orchestrator = get_orchestrator()
    orchestrator.add_documents(chunks)

    return IngestResponse(
        message=f"Successfully ingested directory: {request.directory}",
        chunks_created=len(chunks),
        source=request.directory,
        department=request.department,
    )


# ──────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────

@router.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    current_user: TokenData = Depends(get_current_user),
):
    """Submit feedback on a response (1–5 star rating)."""
    memory = get_memory_manager()
    await memory.save_feedback(
        session_id=request.session_id,
        query=request.query,
        answer=request.answer,
        rating=request.rating,
        helpful=request.helpful,
        comment=request.comment,
    )
    return FeedbackResponse(
        message="Feedback recorded. Thank you!",
        session_id=request.session_id,
    )


# ──────────────────────────────────────────────
# Health & Metadata
# ──────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint. No authentication required."""
    try:
        orchestrator = get_orchestrator()
        store = orchestrator.vector_store
        total_chunks = len(store.chunks) if hasattr(store, "chunks") else -1
    except Exception:
        total_chunks = -1

    return HealthResponse(
        status="healthy",
        vector_store=settings.vector_store_type,
        total_chunks=total_chunks,
        llm_model=settings.llm_model,
    )


@router.get("/sources", response_model=SourcesResponse, tags=["System"])
async def list_sources(
    current_user: TokenData = Depends(get_current_user),
):
    """List all ingested document sources."""
    orchestrator = get_orchestrator()
    store = orchestrator.vector_store

    if hasattr(store, "chunks"):
        chunks = store.chunks
        sources = list({c.source for c in chunks if c.source})
        total = len(chunks)
    else:
        sources = []
        total = -1

    return SourcesResponse(sources=sorted(sources), total_chunks=total)


@router.delete("/session/{session_id}", tags=["System"])
async def clear_session(
    session_id: str,
    current_user: TokenData = Depends(get_current_user),
):
    """Clear conversation history for a session."""
    memory = get_memory_manager()
    await memory.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
