"""
Regulation Checker Agent — 수출입 규정 검색 에이전트 (RAG)

ChromaDB 벡터 검색을 통해 관련 규정 문서를 검색합니다.
Mock 모드에서는 키워드 기반 검색으로 동작합니다.
"""
from langchain_core.messages import HumanMessage

from src.state.customs_state import AgentState
from src.tools.rag_tools import get_customs_procedure, search_customs_regulations
from src.tools.tariff_tools import check_import_restrictions


def _get_last_query(messages: list) -> str:
    return next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )


def regulation_node(state: AgentState) -> dict:
    """
    수출입 규정 검색 에이전트 노드 (RAG).

    1. 관련 규정 문서 벡터 검색
    2. HS Code가 있으면 수입 제한 여부 추가 확인
    3. 통관 절차 키워드 감지 시 절차 안내 추가
    """
    agent_results = dict(state.get("agent_results", {}))
    routing_trace = list(state.get("routing_trace", []))

    query = _get_last_query(state["messages"])
    query_lower = query.lower()

    # 1. RAG 검색
    reg_docs = search_customs_regulations.invoke({"query": query})

    result: dict = {"documents": reg_docs}

    # 2. HS Code가 확인된 경우 수입 제한 조회
    hs_info = agent_results.get("hs_code", {})
    hs_code = hs_info.get("hs_code")
    if hs_code and hs_code != "UNKNOWN":
        restriction = check_import_restrictions.invoke({"hs_code": hs_code})
        result["restriction"] = restriction
        agent_results["regulation"] = restriction

    # 3. 통관 절차 문의 감지
    procedure_keywords = {
        "수입통관": ["수입통관", "수입 통관", "통관 절차", "수입 방법"],
        "여행자통관": ["여행자", "개인 반입", "해외직구", "면세"],
        "수출통관": ["수출통관", "수출 방법"],
    }
    for procedure_type, keywords in procedure_keywords.items():
        if any(kw in query_lower for kw in keywords):
            procedure = get_customs_procedure.invoke({"procedure_type": procedure_type})
            result["procedure"] = procedure
            break

    agent_results["regulation_search"] = result

    trace_entry = {
        "step": state.get("iteration_count", 0),
        "agent": "regulation_agent",
        "result": {"doc_count": len(reg_docs), "has_restriction": "restriction" in result},
    }

    return {
        "agent_results": agent_results,
        "routing_trace": routing_trace + [trace_entry],
    }


def general_agent_node(state: AgentState) -> dict:
    """
    일반 문의 에이전트 노드.
    특정 에이전트가 필요 없는 일반 관세 질문에 답변합니다.
    """
    agent_results = dict(state.get("agent_results", {}))
    routing_trace = list(state.get("routing_trace", []))

    agent_results["general"] = {
        "response": (
            "관세청 AI 상담 시스템입니다. HS코드 조회, 관세율 확인, "
            "수출입 규정 등 구체적인 문의를 해주시면 상세히 안내해 드립니다. "
            "긴급 문의는 관세청 콜센터(☎ 125)로 연락하세요."
        )
    }

    trace_entry = {
        "step": state.get("iteration_count", 0),
        "agent": "general_agent",
        "result": "일반 안내 제공",
    }

    return {
        "agent_results": agent_results,
        "routing_trace": routing_trace + [trace_entry],
    }
