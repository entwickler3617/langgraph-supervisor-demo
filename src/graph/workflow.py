"""
LangGraph Supervisor Workflow — 그래프 조립

★ 시니어 포인트 #2: 멀티턴 컨텍스트 관리
  - MemorySaver를 checkpointer로 지정하면 thread_id 기반으로
    대화 상태가 영구적으로 유지됩니다.
  - 같은 session_id(thread_id)로 호출하면 이전 대화가 자동으로
    state.messages에 포함됩니다.
  - 분산 환경 전환: MemorySaver → RedisSaver 교체만으로 가능.

★ 시니어 포인트 #4: Observability
  - 모든 노드 전환이 LangSmith에 자동 트레이싱됩니다.
  - structlog로 상관 ID 포함 구조화 로그를 남깁니다.
"""
import structlog
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.agents.hs_code_agent import hs_code_node
from src.agents.regulation_agent import general_agent_node, regulation_node
from src.agents.supervisor import generate_final_answer, route_supervisor, supervisor_node
from src.agents.tariff_agent import tariff_node
from src.state.customs_state import AgentState

log = structlog.get_logger(__name__)


def _finish_node(state: AgentState) -> dict:
    """FINISH 상태에서 최종 답변을 생성하는 종료 전 노드."""
    return generate_final_answer(state)


def create_customs_graph() -> StateGraph:
    """
    관세 AI 상담 LangGraph 워크플로우를 생성하고 반환합니다.

    그래프 구조:
        START → supervisor → [hs_code / tariff / regulation / general / finish]
        각 에이전트 노드 → supervisor (결과 보고)
        supervisor FINISH → finish_node → END
    """
    graph = StateGraph(AgentState)

    # ── 노드 등록 ────────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("hs_code_agent", hs_code_node)
    graph.add_node("tariff_agent", tariff_node)
    graph.add_node("regulation_agent", regulation_node)
    graph.add_node("general_agent", general_agent_node)
    graph.add_node("finish", _finish_node)

    # ── 엣지 설정 ────────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    # Supervisor → 조건부 라우팅
    graph.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "hs_code_agent": "hs_code_agent",
            "tariff_agent": "tariff_agent",
            "regulation_agent": "regulation_agent",
            "general_agent": "general_agent",
            "FINISH": "finish",
        },
    )

    # 각 에이전트 → Supervisor로 귀환 (결과 보고)
    graph.add_edge("hs_code_agent", "supervisor")
    graph.add_edge("tariff_agent", "supervisor")
    graph.add_edge("regulation_agent", "supervisor")
    graph.add_edge("general_agent", "finish")

    # 최종 답변 노드 → END
    graph.add_edge("finish", END)

    return graph


def compile_graph(checkpointer=None):
    """
    그래프를 컴파일합니다.

    Args:
        checkpointer: LangGraph 체크포인터 (None이면 MemorySaver 사용)

    Returns:
        컴파일된 CompiledGraph 인스턴스
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = create_customs_graph()
    compiled = graph.compile(checkpointer=checkpointer)
    log.info("customs_graph_compiled", checkpointer_type=type(checkpointer).__name__)
    return compiled


# ── 싱글턴 그래프 인스턴스 (API 서버용) ─────────────────────────────────────
_graph_instance = None


def get_graph():
    """앱 전체에서 공유하는 싱글턴 그래프 인스턴스."""
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = compile_graph()
    return _graph_instance


# ── 고수준 실행 함수 ─────────────────────────────────────────────────────────

def run_chat(session_id: str, user_message: str) -> dict:
    """
    단일 턴 채팅을 실행합니다.

    Args:
        session_id: 세션 식별자 (멀티턴 컨텍스트의 키)
        user_message: 사용자 입력 메시지

    Returns:
        {answer, agents_used, routing_trace, turn_count} dict
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": session_id}}

    initial_state: AgentState = {
        "messages": [HumanMessage(content=user_message)],
        "session_id": session_id,
        "user_intent": None,
        "current_agent": None,
        "agent_results": {},
        "iteration_count": 0,
        "routing_trace": [],
    }

    log.info(
        "chat_started",
        session_id=session_id,
        message_preview=user_message[:50],
    )

    result = graph.invoke(initial_state, config=config)

    # 최종 AI 메시지 추출
    from langchain_core.messages import AIMessage

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    answer = ai_messages[-1].content if ai_messages else "답변을 생성할 수 없습니다."

    agents_used = list(
        {
            t["agent"]
            for t in result.get("routing_trace", [])
            if t["agent"] != "supervisor"
        }
    )

    # 전체 히스토리에서 Human 메시지 수 = 턴 수
    from langchain_core.messages import HumanMessage as HM

    turn_count = sum(1 for m in result["messages"] if isinstance(m, HM))

    log.info(
        "chat_completed",
        session_id=session_id,
        agents_used=agents_used,
        turn_count=turn_count,
    )

    return {
        "answer": answer,
        "agents_used": agents_used,
        "routing_trace": result.get("routing_trace", []),
        "turn_count": turn_count,
    }
