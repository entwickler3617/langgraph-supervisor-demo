"""채팅 API 라우터."""
import structlog
from fastapi import APIRouter, HTTPException

from src.api.schemas import ChatRequest, ChatResponse, SessionHistoryResponse
from src.graph.workflow import get_graph, run_chat

router = APIRouter(prefix="/api/v1", tags=["chat"])
log = structlog.get_logger(__name__)


@router.post("/chat", response_model=ChatResponse, summary="멀티턴 관세 상담")
async def chat(request: ChatRequest) -> ChatResponse:
    """
    관세 AI 상담 엔드포인트.

    - `session_id` 를 유지하면 이전 대화 컨텍스트가 자동으로 포함됩니다.
    - Mock 모드(USE_MOCK_LLM=true)에서는 OpenAI API 키 없이 동작합니다.
    """
    try:
        result = run_chat(
            session_id=request.session_id,
            user_message=request.message,
        )
        return ChatResponse(session_id=request.session_id, **result)
    except Exception as exc:
        log.error("chat_error", session_id=request.session_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"처리 중 오류가 발생했습니다: {exc}") from exc


@router.get(
    "/sessions/{session_id}",
    response_model=SessionHistoryResponse,
    summary="세션 히스토리 조회",
)
async def get_session(session_id: str) -> SessionHistoryResponse:
    """세션의 전체 대화 히스토리를 반환합니다."""
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    try:
        state = graph.get_state(config)
    except Exception:
        return SessionHistoryResponse(
            session_id=session_id, messages=[], turn_count=0
        )

    if not state or not state.values:
        return SessionHistoryResponse(
            session_id=session_id, messages=[], turn_count=0
        )

    messages = state.values.get("messages", [])
    serialized = [
        {"role": type(m).__name__.replace("Message", "").lower(), "content": m.content}
        for m in messages
    ]
    turn_count = sum(1 for m in messages if "Human" in type(m).__name__)

    return SessionHistoryResponse(
        session_id=session_id,
        messages=serialized,
        turn_count=turn_count,
    )


@router.delete("/sessions/{session_id}", summary="세션 초기화")
async def clear_session(session_id: str) -> dict:
    """
    세션을 초기화합니다.
    Note: MemorySaver는 Python 프로세스 재시작 시 자동 초기화됩니다.
    """
    log.info("session_cleared", session_id=session_id)
    return {"status": "cleared", "session_id": session_id}
