"""
LangGraph 상태(State) 정의

★ 시니어 포인트: TypedDict + Annotated[list, add_messages] 조합으로
  LangGraph가 messages 리스트를 자동으로 append-only 방식으로 관리합니다.
  `iteration_count`는 무한 루프 방지 안전장치입니다.
"""
from typing import Annotated, Optional

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """그래프 전체에서 공유되는 상태."""

    # 대화 히스토리 — add_messages 리듀서로 누적 관리
    messages: Annotated[list[BaseMessage], add_messages]

    # 세션 식별자 (멀티턴 컨텍스트의 핵심 키)
    session_id: str

    # Supervisor가 분류한 사용자 의도
    user_intent: Optional[str]

    # 현재 실행 중인 에이전트
    current_agent: Optional[str]

    # 각 에이전트 실행 결과 누적
    agent_results: dict

    # ★ 무한 루프 방지: MAX_ITERATIONS 초과 시 강제 종료
    iteration_count: int

    # 라우팅 추적 로그 (클라이언트에 routing_trace로 반환)
    routing_trace: list[dict]
