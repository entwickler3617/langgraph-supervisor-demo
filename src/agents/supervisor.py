"""
Supervisor Agent — 의도 분류 및 동적 라우팅

★ 시니어 포인트 #1: Supervisor 패턴
  - LLM이 매 턴마다 어떤 에이전트를 호출할지 결정합니다.
  - RouterDecision에 'reasoning' 필드를 포함해 라우팅 근거를 기록합니다.
    → 프로덕션 디버깅, 감사(audit trail)에 활용.
  - MAX_ITERATIONS로 무한 루프를 방지합니다.

★ 시니어 포인트 #5: Graceful Degradation
  - USE_MOCK_LLM=true 시 MockSupervisorLLM이 키워드 기반으로 라우팅.
  - 실제 LLM 오류 시 'general_agent'로 폴백.
"""
import re
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from src.config import get_settings
from src.state.customs_state import AgentState

MAX_ITERATIONS = 5

# ── Structured Output 모델 ──────────────────────────────────────────────────

class RouterDecision(BaseModel):
    """Supervisor LLM의 라우팅 결정."""

    next: Literal["hs_code_agent", "tariff_agent", "regulation_agent", "general_agent", "FINISH"] = Field(
        description="다음에 호출할 에이전트 이름 또는 FINISH"
    )
    reasoning: str = Field(description="라우팅 결정 근거 (감사 로그용)")
    requires_more_info: bool = Field(
        default=False, description="추가 사용자 입력이 필요한지 여부"
    )


# ── Mock LLM (API 키 없이 동작) ─────────────────────────────────────────────

class MockSupervisorLLM:
    """
    ★ Graceful Degradation: 결정론적 키워드 기반 라우팅 Mock LLM.
    CI/CD 환경 및 데모 시연에서 비용 없이 동작합니다.
    """

    _INTENT_KEYWORDS = {
        "hs_code_agent": [
            "hs코드", "hs code", "품목번호", "품목 번호", "분류", "몇 번",
        ],
        "tariff_agent": [
            "관세율", "세율", "관세", "tariff", "얼마", "요율", "세금",
        ],
        "regulation_agent": [
            "규정", "규제", "금지", "제한", "허용", "통관", "절차", "서류",
            "면세", "신고", "반입",
        ],
    }

    def invoke(self, messages: list, agent_results: dict) -> RouterDecision:
        last_human = next(
            (m.content for m in reversed(messages) if isinstance(m, HumanMessage)),
            "",
        )
        text = last_human.lower()

        # 이미 수집한 결과를 기반으로 다음 단계 결정
        if "hs_code" not in agent_results and any(
            kw in text for kw in self._INTENT_KEYWORDS["tariff_agent"] + self._INTENT_KEYWORDS["hs_code_agent"]
        ):
            return RouterDecision(
                next="hs_code_agent",
                reasoning="관세율 조회 전 HS Code 확인이 필요합니다.",
            )

        if "hs_code" in agent_results and "tariff_rate" not in agent_results and any(
            kw in text for kw in self._INTENT_KEYWORDS["tariff_agent"]
        ):
            return RouterDecision(
                next="tariff_agent",
                reasoning="HS Code 확인 완료. 관세율 조회를 진행합니다.",
            )

        if any(kw in text for kw in self._INTENT_KEYWORDS["regulation_agent"]):
            return RouterDecision(
                next="regulation_agent",
                reasoning="수출입 규정 관련 질문입니다.",
            )

        if agent_results:
            return RouterDecision(
                next="FINISH",
                reasoning="필요한 정보를 모두 수집했습니다. 최종 답변을 생성합니다.",
            )

        return RouterDecision(
            next="general_agent",
            reasoning="특정 에이전트가 필요 없는 일반 문의입니다.",
        )


# ── Real LLM Supervisor ─────────────────────────────────────────────────────

SUPERVISOR_SYSTEM_PROMPT = """당신은 관세청 AI 상담 시스템의 Supervisor입니다.
사용자 질문을 분석하고, 다음 중 가장 적합한 에이전트를 선택하세요:

- hs_code_agent: 상품 HS Code(품목번호) 분류가 필요한 경우
- tariff_agent: 관세율, FTA 세율 조회가 필요한 경우  
- regulation_agent: 수출입 규정, 통관 절차, 면세 한도 조회가 필요한 경우
- general_agent: 위에 해당하지 않는 일반 관세 문의
- FINISH: 모든 필요한 정보가 수집되어 최종 답변 생성 가능

이미 수집된 결과: {agent_results}
반복 횟수: {iteration}/{max_iterations}
"""


def _build_real_supervisor(settings):
    """실제 OpenAI LLM을 사용하는 Supervisor 생성."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )
    return llm.with_structured_output(RouterDecision)


# ── 그래프 노드 함수 ─────────────────────────────────────────────────────────

def supervisor_node(state: AgentState) -> dict:
    """
    Supervisor 노드: 의도 분류 후 다음 에이전트를 결정합니다.

    Returns:
        상태 업데이트 dict (current_agent, routing_trace, iteration_count 포함)
    """
    settings = get_settings()
    iteration = state.get("iteration_count", 0) + 1
    agent_results = state.get("agent_results", {})
    routing_trace = state.get("routing_trace", [])

    # 무한 루프 안전장치
    if iteration > MAX_ITERATIONS:
        trace_entry = {
            "step": iteration,
            "agent": "supervisor",
            "decision": "FINISH",
            "reasoning": f"최대 반복 횟수({MAX_ITERATIONS}) 초과. 강제 종료.",
        }
        return {
            "current_agent": "FINISH",
            "iteration_count": iteration,
            "routing_trace": routing_trace + [trace_entry],
        }

    # 라우팅 결정
    try:
        if settings.use_mock_llm:
            mock_llm = MockSupervisorLLM()
            decision = mock_llm.invoke(state["messages"], agent_results)
        else:
            real_llm = _build_real_supervisor(settings)
            system_prompt = SUPERVISOR_SYSTEM_PROMPT.format(
                agent_results=agent_results,
                iteration=iteration,
                max_iterations=MAX_ITERATIONS,
            )
            messages_with_system = [SystemMessage(content=system_prompt)] + state["messages"]
            decision = real_llm.invoke(messages_with_system)
    except Exception as exc:
        # ★ Graceful Degradation: LLM 오류 시 일반 에이전트로 폴백
        decision = RouterDecision(
            next="general_agent",
            reasoning=f"LLM 오류로 general_agent로 폴백: {exc}",
        )

    trace_entry = {
        "step": iteration,
        "agent": "supervisor",
        "decision": decision.next,
        "reasoning": decision.reasoning,
    }

    return {
        "current_agent": decision.next,
        "iteration_count": iteration,
        "routing_trace": routing_trace + [trace_entry],
    }


def route_supervisor(state: AgentState) -> str:
    """조건부 엣지: supervisor 결정에 따라 다음 노드 반환."""
    return state.get("current_agent", "FINISH")


def generate_final_answer(state: AgentState) -> dict:
    """
    모든 에이전트 결과를 통합해 최종 자연어 답변을 생성합니다.
    Supervisor가 FINISH 결정 시 호출됩니다.
    """
    settings = get_settings()
    agent_results = state.get("agent_results", {})
    messages = state["messages"]

    last_human = next(
        (m.content for m in reversed(messages) if isinstance(m, HumanMessage)), ""
    )

    if settings.use_mock_llm:
        answer = _build_mock_answer(last_human, agent_results)
    else:
        answer = _build_llm_answer(last_human, agent_results, settings)

    return {"messages": [AIMessage(content=answer)]}


def _build_mock_answer(query: str, results: dict) -> str:
    """Mock 최종 답변 생성."""
    parts = [f"**관세 AI 상담 결과** (질문: {query})\n"]

    if hs := results.get("hs_code"):
        parts.append(f"📦 **HS Code**: {hs.get('hs_code')} — {hs.get('description')}")
        parts.append(f"   (분류 신뢰도: {hs.get('confidence', 0):.0%})")

    if tr := results.get("tariff_rate"):
        parts.append(f"\n💰 **관세율**: 기본세율 {tr.get('basic_rate')}")
        if tr.get("fta_rate"):
            parts.append(f"   FTA 협정세율({tr.get('fta_agreement')}): {tr.get('fta_rate')}")
        if tr.get("special_taxes"):
            parts.append(f"   기타 세금: {', '.join(tr['special_taxes'])}")

    if reg := results.get("regulation"):
        status_emoji = {"ALLOWED": "✅", "RESTRICTED": "⚠️", "PROHIBITED": "❌"}.get(
            reg.get("status"), "ℹ️"
        )
        parts.append(f"\n{status_emoji} **수입 가능 여부**: {reg.get('status')} — {reg.get('reason')}")

    if not results:
        parts.append("안녕하세요! 관세율, HS코드, 수출입 규정 등 관세 관련 문의를 도와드립니다.")

    parts.append("\n> ⚠️ 본 답변은 참고용입니다. 실제 신고 시 관세사 확인을 권장합니다.")
    return "\n".join(parts)


def _build_llm_answer(query: str, results: dict, settings) -> str:
    """실제 LLM으로 최종 답변 생성."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=settings.openai_model, api_key=settings.openai_api_key)
    prompt = f"""다음 관세 조회 결과를 바탕으로 사용자 질문에 친절하고 명확하게 답변하세요.

사용자 질문: {query}
조회 결과: {results}

- 한국어로 답변하세요.
- 중요 수치(관세율, HS코드)는 강조 표시하세요.
- 마지막에 "실제 신고 시 관세사 확인을 권장합니다" 문구를 추가하세요.
"""
    response = llm.invoke(prompt)
    return response.content
