"""
Tariff Rate Lookup Agent — 관세율 조회 에이전트

이전 단계에서 수집된 HS Code를 기반으로 관세율을 조회합니다.
FTA 협정세율과 특별소비세 등 복합 세율 정보를 제공합니다.
"""
import re

from langchain_core.messages import HumanMessage

from src.state.customs_state import AgentState
from src.tools.tariff_tools import get_tariff_rate


def _extract_origin_country(messages: list) -> str:
    """대화에서 원산지 국가를 추출합니다."""
    country_patterns = {
        "미국": r"미국|us|america|미제",
        "중국": r"중국|china|중제",
        "유럽": r"유럽|eu|europe|독일|프랑스|이탈리아",
        "일본": r"일본|japan|일제",
    }
    combined_text = " ".join(
        m.content for m in messages if isinstance(m, HumanMessage)
    ).lower()

    for country, pattern in country_patterns.items():
        if re.search(pattern, combined_text):
            return country

    return "일반"  # 원산지 불명 시 기본세율 반환


def tariff_node(state: AgentState) -> dict:
    """
    관세율 조회 에이전트 노드.

    1. 이전 단계의 HS Code 결과 사용 (없으면 에러 처리)
    2. 대화에서 원산지 국가 추출
    3. get_tariff_rate 툴 호출
    4. agent_results에 결과 저장
    """
    agent_results = dict(state.get("agent_results", {}))
    routing_trace = list(state.get("routing_trace", []))

    # 이전 단계 HS Code 참조
    hs_info = agent_results.get("hs_code", {})
    hs_code = hs_info.get("hs_code", "UNKNOWN")

    if hs_code == "UNKNOWN":
        result = {"error": "HS Code가 확인되지 않아 관세율을 조회할 수 없습니다."}
    else:
        origin = _extract_origin_country(state["messages"])
        result = get_tariff_rate.invoke({"hs_code": hs_code, "origin_country": origin})

    agent_results["tariff_rate"] = result

    trace_entry = {
        "step": state.get("iteration_count", 0),
        "agent": "tariff_agent",
        "result": result,
    }

    return {
        "agent_results": agent_results,
        "routing_trace": routing_trace + [trace_entry],
    }
