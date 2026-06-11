# api/models.py
"""
Step 10: Pydantic Request/Response Schemas
Clear, validated API contracts for all endpoints.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────
# Auth
# ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str = Field(..., example="engineer")
    password: str = Field(..., example="eng123")


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    roles: List[str]


# ──────────────────────────────────────────────
# Query
# ──────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="What is the policy for remote work?"
    )
    session_id: str = Field(
        default="default",
        description="Session ID for conversation memory"
    )
    use_history: bool = Field(
        default=True,
        description="Whether to include conversation history in context"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the onboarding process for new employees?",
                "session_id": "sess_abc123",
                "use_history": True,
            }
        }


class CitationModel(BaseModel):
    source: str
    page_number: Optional[int]
    section: Optional[str]
    snippet: str
    citation_str: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[CitationModel]
    confidence: float = Field(ge=0.0, le=1.0)
    query: str
    sources: List[str]
    steps_completed: List[str] = []
    error: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The remote work policy allows up to 3 days per week. [SOURCE_1]",
                "citations": [
                    {
                        "source": "hr_policy_2024.pdf",
                        "page_number": 12,
                        "section": "Remote Work",
                        "snippet": "Employees may work remotely up to 3 days per week...",
                        "citation_str": "[hr_policy_2024.pdf, p.12, §Remote Work]",
                    }
                ],
                "confidence": 0.87,
                "query": "What is the remote work policy?",
                "sources": ["hr_policy_2024.pdf"],
                "steps_completed": ["retrieval", "validation", "response"],
            }
        }


# ──────────────────────────────────────────────
# Document Ingestion
# ──────────────────────────────────────────────

class IngestResponse(BaseModel):
    message: str
    chunks_created: int
    source: str
    department: Optional[str]


class IngestDirectoryRequest(BaseModel):
    directory: str = Field(..., example="./data/documents/hr")
    department: Optional[str] = Field(None, example="hr")
    access_roles: Optional[List[str]] = Field(
        default=["all"], example=["hr", "admin"]
    )


# ──────────────────────────────────────────────
# Feedback
# ──────────────────────────────────────────────

class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    answer: str
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 (bad) to 5 (excellent)")
    helpful: Optional[bool] = None
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    message: str
    session_id: str


# ──────────────────────────────────────────────
# Health & Sources
# ──────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store: str
    total_chunks: int
    llm_model: str
    version: str = "1.0.0"


class SourcesResponse(BaseModel):
    sources: List[str]
    total_chunks: int
