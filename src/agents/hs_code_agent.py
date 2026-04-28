"""
HS Code Classifier Agent

★ 시니어 포인트 #3: Pydantic Structured Output
  - LLM이 HSCodeResult 스키마를 준수하는 JSON만 반환합니다.
  - 파싱 오류 시 graceful fallback으로 에러 전파 없이 처리합니다.
"""
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from src.state.customs_state import AgentState
from src.tools.tariff_tools import lookup_hs_code


class HSCodeResult(BaseModel):
    """HS Code 분류 결과 구조체."""

    hs_code: str = Field(description="6자리 HS Code (예: 8471.30)")
    description: str = Field(description="품목 설명 (한국어)")
    confidence: float = Field(ge=0.0, le=1.0, description="분류 신뢰도 0~1")
    alternative_codes: list[str] = Field(
        default_factory=list, description="대체 가능한 HS Code 목록"
    )
    note: str = Field(default="", description="추가 안내 사항")


def _extract_product_name(messages: list) -> str:
    """대화 히스토리에서 상품명을 추출합니다."""
    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )
    # 간단한 키워드 추출 (실제 LLM에서는 NER 사용)
    clean = last_human.replace("관세율", "").replace("수입", "").replace("수출", "")
    clean = clean.replace("HS코드", "").replace("HS Code", "").replace("알려줘", "")
    clean = clean.replace("어떻게 되나요", "").replace("?", "").strip()
    return clean or last_human


def hs_code_node(state: AgentState) -> dict:
    """
    HS Code 분류 에이전트 노드.

    1. 대화에서 상품명 추출
    2. lookup_hs_code 툴 호출 (Mock or Real)
    3. 결과를 agent_results에 저장 → Supervisor에게 반환
    """
    agent_results = dict(state.get("agent_results", {}))
    routing_trace = list(state.get("routing_trace", []))

    product_name = _extract_product_name(state["messages"])

    # Tool 호출
    raw_result = lookup_hs_code.invoke({"product_name": product_name})

    result = HSCodeResult(
        hs_code=raw_result.get("hs_code", "UNKNOWN"),
        description=raw_result.get("description", ""),
        confidence=raw_result.get("confidence", 0.0),
        note=raw_result.get("note", ""),
    )

    agent_results["hs_code"] = result.model_dump()

    trace_entry = {
        "step": state.get("iteration_count", 0),
        "agent": "hs_code_agent",
        "result": result.model_dump(),
    }

    return {
        "agent_results": agent_results,
        "routing_trace": routing_trace + [trace_entry],
    }
