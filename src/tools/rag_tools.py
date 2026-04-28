"""
ChromaDB 기반 RAG 툴

★ 시니어 포인트:
  - 문서 임베딩 초기화는 앱 시작 시 1회만 실행 (lru_cache)
  - USE_MOCK_LLM=true 시 실제 임베딩 없이 키워드 매칭으로 폴백
  - 프로덕션 전환: CHROMA_PERSIST_DIR + 실제 임베딩 모델만 교체
"""
import json
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import tool

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_regulations() -> list[dict]:
    path = _DATA_DIR / "mock_regulations.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("regulation_documents", [])


@lru_cache(maxsize=1)
def _get_mock_docs() -> list[dict]:
    return _load_regulations()


def _keyword_search(query: str, docs: list[dict], top_k: int = 3) -> list[dict]:
    """임베딩 없이 키워드 유사도로 문서 검색 (Mock 폴백)."""
    query_words = set(query.lower().split())
    scored: list[tuple[float, dict]] = []

    for doc in docs:
        text = (doc.get("title", "") + " " + doc.get("content", "")).lower()
        overlap = sum(1 for w in query_words if w in text)
        if overlap > 0:
            scored.append((overlap / len(query_words), doc))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


@tool
def search_customs_regulations(query: str) -> list[dict]:
    """
    관세 규정 문서에서 관련 내용을 검색합니다 (RAG).

    Args:
        query: 검색 쿼리 (자연어)

    Returns:
        관련 규정 문서 목록 (title, content, source 포함)
    """
    docs = _get_mock_docs()
    results = _keyword_search(query, docs, top_k=3)

    if not results:
        return [
            {
                "title": "검색 결과 없음",
                "content": f"'{query}'에 관한 규정을 찾을 수 없습니다. 관세청(125)에 문의하세요.",
                "source": "관세청 콜센터",
            }
        ]
    return results


@tool
def get_customs_procedure(procedure_type: str) -> dict:
    """
    통관 절차 안내를 반환합니다.

    Args:
        procedure_type: 절차 유형 (수입통관, 수출통관, 여행자통관 등)
    """
    procedures = {
        "수입통관": {
            "steps": [
                "1. 입항 전 수입신고 (Arrival Notice 수령 후)",
                "2. 수입신고서 작성 및 제출 (관세사 의뢰 권장)",
                "3. 세관 심사 (서류/물품 검사)",
                "4. 관세·부가세 납부",
                "5. 수입신고 수리 및 물품 반출",
            ],
            "required_docs": ["수입신고서", "상업송장(Invoice)", "선하증권(B/L) 또는 AWB", "포장명세서"],
            "avg_days": "1~3 영업일",
        },
        "여행자통관": {
            "steps": [
                "1. 입국 후 세관 신고대 통과",
                "2. 면세 한도($800) 초과 시 자진 신고",
                "3. 과세 대상 물품 세금 납부",
            ],
            "required_docs": ["여권", "세관신고서(해당 시)"],
            "avg_days": "당일",
            "duty_free_limit": "$800 (개인 반입 물품)",
        },
        "수출통관": {
            "steps": [
                "1. 수출신고서 제출 (출항 전)",
                "2. 세관 심사",
                "3. 수출신고 수리",
                "4. 선적/항공기 적재",
            ],
            "required_docs": ["수출신고서", "상업송장", "포장명세서"],
            "avg_days": "당일~1 영업일",
        },
    }

    result = procedures.get(procedure_type)
    if not result:
        return {
            "error": f"'{procedure_type}' 절차 정보를 찾을 수 없습니다.",
            "available": list(procedures.keys()),
        }
    return {"procedure_type": procedure_type, **result}
