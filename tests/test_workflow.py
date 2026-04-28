"""
LangGraph 워크플로우 통합 테스트

★ 시니어 포인트: Mock 모드에서 전체 에이전트 그래프 통합 테스트.
  - API 키 없이 CI/CD에서 실행 가능
  - 멀티턴 컨텍스트 유지 테스트 포함
"""
import pytest

from src.graph.workflow import compile_graph, run_chat


@pytest.fixture
def graph():
    """각 테스트마다 새 그래프 인스턴스 (독립 메모리)."""
    return compile_graph()


class TestSingleTurnChat:
    def test_laptop_tariff_query(self):
        result = run_chat(
            session_id="test-laptop",
            user_message="노트북 미국에서 수입할 때 관세율이 어떻게 되나요?",
        )
        assert result["answer"] != ""
        assert result["turn_count"] >= 1
        assert isinstance(result["agents_used"], list)
        assert isinstance(result["routing_trace"], list)
        assert len(result["routing_trace"]) > 0

    def test_regulation_query(self):
        result = run_chat(
            session_id="test-regulation",
            user_message="해외직구 면세 한도가 얼마인가요?",
        )
        assert result["answer"] != ""
        assert "regulation_agent" in result["agents_used"]

    def test_general_query(self):
        result = run_chat(
            session_id="test-general",
            user_message="관세청 전화번호가 뭐예요?",
        )
        assert result["answer"] != ""

    def test_routing_trace_has_supervisor(self):
        result = run_chat(
            session_id="test-trace",
            user_message="노트북 관세율",
        )
        agents_in_trace = {t["agent"] for t in result["routing_trace"]}
        assert "supervisor" in agents_in_trace

    def test_answer_contains_customs_info(self):
        result = run_chat(
            session_id="test-content",
            user_message="노트북 관세율 알려줘",
        )
        # Mock 답변에 HS코드 또는 관세율 정보 포함
        answer = result["answer"]
        assert any(
            kw in answer for kw in ["HS", "관세", "세율", "8471"]
        ), f"Expected customs info in answer, got: {answer[:200]}"


class TestMultiTurnConversation:
    """
    ★ 멀티턴 컨텍스트 유지 테스트
    같은 session_id로 연속 호출 시 이전 대화가 유지되어야 합니다.
    """

    def test_turn_count_increments(self):
        session = "test-multiturn-001"

        result1 = run_chat(session_id=session, user_message="노트북 관세율 알려줘")
        assert result1["turn_count"] == 1

        result2 = run_chat(session_id=session, user_message="그러면 일본산은요?")
        assert result2["turn_count"] == 2

    def test_different_sessions_are_independent(self):
        result_a = run_chat(session_id="session-a", user_message="노트북 관세율")
        result_b = run_chat(session_id="session-b", user_message="맥주 관세율")

        # 각 세션은 독립적
        assert result_a["turn_count"] == 1
        assert result_b["turn_count"] == 1


class TestSafetyGuards:
    def test_handles_empty_like_query(self):
        """빈 의미의 질문도 에러 없이 처리되어야 합니다."""
        result = run_chat(
            session_id="test-empty",
            user_message="ㅎㅎ",
        )
        assert "answer" in result

    def test_iteration_limit_not_exceeded(self):
        """최대 반복 횟수를 초과하지 않아야 합니다."""
        result = run_chat(
            session_id="test-iter",
            user_message="노트북 미국에서 수입 관세율 알려줘",
        )
        iterations = max(
            (t.get("step", 0) for t in result["routing_trace"]),
            default=0,
        )
        assert iterations <= 5, f"Exceeded max iterations: {iterations}"
