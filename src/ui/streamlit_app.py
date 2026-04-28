"""
Streamlit 데모 UI — 관세청 AI 상담 챗봇

실행 방법:
    streamlit run src/ui/streamlit_app.py
    # 또는
    make ui
"""
import json
import sys
from pathlib import Path

import streamlit as st

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.graph.workflow import run_chat

# ── 페이지 설정 ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="관세청 AI 상담 시스템",
    page_icon="🛃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS 커스터마이징 ──────────────────────────────────────────────────────────
st.markdown(
    """
<style>
.routing-trace {
    background: #f8f9fa;
    border-left: 3px solid #2196F3;
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.85rem;
}
.agent-badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
    margin: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── 사이드바 ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🛃 관세청 AI 상담")
    st.markdown("---")

    st.subheader("⚙️ 설정")
    session_id = st.text_input(
        "세션 ID",
        value="demo-session-001",
        help="같은 세션 ID를 유지하면 이전 대화 컨텍스트가 유지됩니다.",
    )

    if st.button("🗑️ 대화 초기화", use_container_width=True):
        st.session_state.messages = []
        st.session_state.trace_history = []
        st.rerun()

    st.markdown("---")
    st.subheader("💡 예시 질문")
    examples = [
        "노트북 미국에서 수입 관세율?",
        "아이폰 케이스 HS코드 알려줘",
        "쇠고기 수입 제한 있나요?",
        "해외직구 면세 한도가 얼마야?",
        "수입통관 절차 설명해줘",
        "맥주 중국에서 수입하면 관세 얼마야?",
    ]
    for example in examples:
        if st.button(f"📌 {example}", use_container_width=True, key=f"ex_{example}"):
            st.session_state.pending_input = example

    st.markdown("---")
    st.caption("🔧 Mock 모드로 동작 중 (USE_MOCK_LLM=true)")
    st.caption("실제 OpenAI 연동은 .env 설정 참고")

# ── 메인 화면 ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.title("🤖 관세청 AI 상담 시스템")
    st.markdown(
        "**LangGraph Supervisor Multi-Agent** — HS코드 분류 · 관세율 조회 · 수출입 규정 안내"
    )
    st.markdown("---")

    # 대화 히스토리 초기화
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "trace_history" not in st.session_state:
        st.session_state.trace_history = []

    # 대화 표시
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 예시 버튼에서 온 입력 처리
    prefill = st.session_state.pop("pending_input", None)

    # 입력창
    user_input = st.chat_input("관세 관련 질문을 입력하세요...") or prefill

    if user_input:
        # 사용자 메시지 표시
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # 에이전트 실행
        with st.spinner("🔄 에이전트 분석 중..."):
            try:
                result = run_chat(
                    session_id=session_id,
                    user_message=user_input,
                )
                answer = result["answer"]
                trace = result["routing_trace"]
                agents_used = result["agents_used"]
            except Exception as e:
                answer = f"❌ 오류가 발생했습니다: {e}"
                trace = []
                agents_used = []

        # AI 응답 표시
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.trace_history.append(
            {"query": user_input, "trace": trace, "agents": agents_used}
        )
        st.rerun()

with col2:
    st.subheader("🔍 에이전트 실행 추적")
    st.markdown("*Supervisor가 어떤 에이전트를 호출했는지 실시간으로 확인합니다.*")

    if st.session_state.get("trace_history"):
        latest = st.session_state.trace_history[-1]

        # 사용된 에이전트 배지
        agent_colors = {
            "hs_code_agent": "#4CAF50",
            "tariff_agent": "#2196F3",
            "regulation_agent": "#FF9800",
            "general_agent": "#9C27B0",
            "finish": "#607D8B",
        }
        if latest["agents"]:
            badges = " ".join(
                f'<span class="agent-badge" style="background:{agent_colors.get(a, "#666")};color:white">'
                f"{a}</span>"
                for a in latest["agents"]
            )
            st.markdown(f"**호출된 에이전트**: {badges}", unsafe_allow_html=True)

        # 라우팅 트레이스
        st.markdown("**실행 흐름:**")
        for step in latest["trace"]:
            icon = {
                "supervisor": "🎯",
                "hs_code_agent": "🏷️",
                "tariff_agent": "💰",
                "regulation_agent": "📋",
                "general_agent": "💬",
                "finish": "✅",
            }.get(step["agent"], "⚙️")

            if step.get("decision"):
                st.markdown(
                    f'<div class="routing-trace">'
                    f"{icon} <b>{step['agent']}</b> → {step['decision']}<br>"
                    f"<small>{step.get('reasoning', '')}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif step.get("result"):
                result_str = json.dumps(step["result"], ensure_ascii=False, indent=2)
                st.markdown(
                    f'<div class="routing-trace">'
                    f"{icon} <b>{step['agent']}</b> 결과<br>"
                    f"<small><pre>{result_str[:200]}{'...' if len(result_str) > 200 else ''}</pre></small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # 전체 히스토리 토글
        if len(st.session_state.trace_history) > 1:
            with st.expander(f"이전 {len(st.session_state.trace_history) - 1}개 추적 로그"):
                for i, hist in enumerate(st.session_state.trace_history[:-1]):
                    st.markdown(f"**Q{i+1}**: {hist['query'][:50]}...")
    else:
        st.info("질문을 입력하면 에이전트 실행 흐름이 여기에 표시됩니다.")

    # 아키텍처 설명
    st.markdown("---")
    st.subheader("🏗️ 아키텍처")
    st.markdown(
        """
```
사용자 질문
    ↓
🎯 Supervisor (의도 분류)
    ├── 🏷️ HS Code Agent
    ├── 💰 Tariff Agent  
    ├── 📋 Regulation Agent (RAG)
    └── 💬 General Agent
    ↓
✅ 최종 답변 통합
```
*모든 대화가 세션 메모리에 저장되어 멀티턴 컨텍스트가 유지됩니다.*
"""
    )
