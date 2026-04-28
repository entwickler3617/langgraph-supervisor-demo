"""FastAPI Request/Response 스키마 — Pydantic v2."""
from typing import Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="세션 식별자 (멀티턴 컨텍스트 키)",
        examples=["user-abc-123"],
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="사용자 메시지",
        examples=["노트북 미국에서 수입할 때 관세율이 어떻게 되나요?"],
    )


class RoutingTraceItem(BaseModel):
    step: int
    agent: str
    decision: Optional[str] = None
    reasoning: Optional[str] = None
    result: Optional[dict] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    agents_used: list[str]
    routing_trace: list[dict]
    turn_count: int


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: list[dict]
    turn_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    mock_mode: bool
