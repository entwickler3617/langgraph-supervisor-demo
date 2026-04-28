"""
에이전트 단위 테스트

★ 시니어 포인트: Mock LLM 모드에서 외부 API 없이 CI/CD 파이프라인에서 실행 가능.
"""
import pytest

from src.agents.hs_code_agent import hs_code_node
from src.agents.regulation_agent import general_agent_node, regulation_node
from src.agents.supervisor import MockSupervisorLLM, RouterDecision, supervisor_node
from src.agents.tariff_agent import tariff_node
from src.state.customs_state import AgentState
from langchain_core.messages import HumanMessage


def _make_state(message: str, agent_results: dict = None) -> AgentState:
    """테스트용 AgentState 팩토리."""
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": "test-session",
        "user_intent": None,
        "current_agent": None,
        "agent_results": agent_results or {},
        "iteration_count": 0,
        "routing_trace": [],
    }


# ── Supervisor Tests ─────────────────────────────────────────────────────────

class TestMockSupervisorLLM:
    def setup_method(self):
        self.llm = MockSupervisorLLM()

    def test_routes_to_hs_code_agent_for_tariff_query(self):
        messages = [HumanMessage(content="노트북 관세율 알고 싶어요")]
        decision = self.llm.invoke(messages, agent_results={})
        assert decision.next == "hs_code_agent"
        assert decision.reasoning != ""

    def test_routes_to_tariff_agent_when_hs_known(self):
        messages = [HumanMessage(content="관세율 알려줘")]
        agent_results = {"hs_code": {"hs_code": "8471.30"}}
        decision = self.llm.invoke(messages, agent_results=agent_results)
        assert decision.next == "tariff_agent"

    def test_routes_to_regulation_agent(self):
        messages = [HumanMessage(content="수입통관 절차가 궁금해요")]
        decision = self.llm.invoke(messages, agent_results={})
        assert decision.next == "regulation_agent"

    def test_finishes_when_results_collected(self):
        messages = [HumanMessage(content="관세율 알려줘")]
        agent_results = {
            "hs_code": {"hs_code": "8471.30"},
            "tariff_rate": {"basic_rate": "0%"},
        }
        decision = self.llm.invoke(messages, agent_results=agent_results)
        assert decision.next == "FINISH"

    def test_routes_to_general_for_unknown_intent(self):
        messages = [HumanMessage(content="안녕하세요")]
        decision = self.llm.invoke(messages, agent_results={})
        assert decision.next == "general_agent"


class TestSupervisorNode:
    def test_supervisor_increments_iteration(self):
        state = _make_state("노트북 관세율")
        result = supervisor_node(state)
        assert result["iteration_count"] == 1

    def test_supervisor_adds_trace(self):
        state = _make_state("노트북 관세율")
        result = supervisor_node(state)
        assert len(result["routing_trace"]) == 1
        assert result["routing_trace"][0]["agent"] == "supervisor"

    def test_supervisor_force_finish_at_max_iterations(self):
        state = _make_state("질문")
        state["iteration_count"] = 5  # MAX_ITERATIONS
        result = supervisor_node(state)
        assert result["current_agent"] == "FINISH"


# ── HS Code Agent Tests ──────────────────────────────────────────────────────

class TestHSCodeAgent:
    def test_classifies_laptop(self):
        state = _make_state("노트북 수입 관세")
        result = hs_code_node(state)
        hs = result["agent_results"]["hs_code"]
        assert hs["hs_code"] == "8471.30"
        assert hs["confidence"] > 0.5

    def test_classifies_smartphone(self):
        state = _make_state("스마트폰 관세율")
        result = hs_code_node(state)
        hs = result["agent_results"]["hs_code"]
        assert hs["hs_code"] == "8517.12"

    def test_returns_unknown_for_unrecognized_product(self):
        state = _make_state("매우 특이한 제품명 XYZ")
        result = hs_code_node(state)
        hs = result["agent_results"]["hs_code"]
        assert hs["hs_code"] == "UNKNOWN"
        assert hs["confidence"] == 0.0

    def test_adds_trace_entry(self):
        state = _make_state("노트북")
        result = hs_code_node(state)
        assert len(result["routing_trace"]) == 1


# ── Tariff Agent Tests ───────────────────────────────────────────────────────

class TestTariffAgent:
    def test_returns_tariff_for_known_hs_code(self):
        state = _make_state(
            "미국산 노트북 관세율",
            agent_results={"hs_code": {"hs_code": "8471.30"}},
        )
        result = tariff_node(state)
        tr = result["agent_results"]["tariff_rate"]
        assert tr["basic_rate"] == "0%"
        assert tr["fta_rate"] == "0%"

    def test_returns_error_without_hs_code(self):
        state = _make_state("관세율 알려줘", agent_results={})
        result = tariff_node(state)
        tr = result["agent_results"]["tariff_rate"]
        assert "error" in tr

    def test_extracts_origin_country_from_message(self):
        state = _make_state(
            "중국에서 맥주 수입하면 관세 얼마야?",
            agent_results={"hs_code": {"hs_code": "2203.00"}},
        )
        result = tariff_node(state)
        tr = result["agent_results"]["tariff_rate"]
        assert tr["origin_country"] == "중국"


# ── Regulation Agent Tests ───────────────────────────────────────────────────

class TestRegulationAgent:
    def test_searches_regulations(self):
        state = _make_state("면세 한도가 얼마야")
        result = regulation_node(state)
        reg = result["agent_results"]["regulation_search"]
        assert "documents" in reg
        assert len(reg["documents"]) > 0

    def test_includes_restriction_when_hs_known(self):
        state = _make_state(
            "쇠고기 수입 가능해?",
            agent_results={"hs_code": {"hs_code": "0201.10"}},
        )
        result = regulation_node(state)
        assert "regulation" in result["agent_results"]

    def test_includes_procedure_for_customs_query(self):
        state = _make_state("수입통관 절차 알려줘")
        result = regulation_node(state)
        reg = result["agent_results"]["regulation_search"]
        assert "procedure" in reg

    def test_general_agent_returns_help_message(self):
        state = _make_state("안녕하세요")
        result = general_agent_node(state)
        assert "general" in result["agent_results"]
        assert "관세" in result["agent_results"]["general"]["response"]
