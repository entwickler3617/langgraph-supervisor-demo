"""
관세 도메인 Mock 데이터 + Tool 함수

★ 시니어 포인트 (Graceful Degradation):
  - 모든 툴은 Mock 데이터로 동작하므로 외부 API 없이 전체 데모가 실행됩니다.
  - 실제 관세청 Open API 연동은 이 파일의 Mock만 교체하면 됩니다.
"""
import json
from pathlib import Path

from langchain_core.tools import tool

# ── Mock 데이터 로드 ────────────────────────────────────────────────────────
_DATA_DIR = Path(__file__).parent.parent.parent / "data"


def _load_json(filename: str) -> dict:
    path = _DATA_DIR / filename
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


_TARIFF_DATA: dict = _load_json("mock_tariff_data.json")
_REGULATION_DATA: dict = _load_json("mock_regulations.json")


# ── Tool 정의 ───────────────────────────────────────────────────────────────

@tool
def lookup_hs_code(product_name: str) -> dict:
    """
    상품명으로 HS Code를 조회합니다.

    Args:
        product_name: 조회할 상품명 (한국어 또는 영어)

    Returns:
        hs_code, description, confidence, note 포함 dict
    """
    name_lower = product_name.lower()
    hs_db = _TARIFF_DATA.get("hs_codes", {})

    # 키워드 매칭 (Mock 로직)
    keyword_map = {
        ("노트북", "laptop", "컴퓨터"): "8471.30",
        ("스마트폰", "핸드폰", "휴대폰", "iphone", "아이폰"): "8517.12",
        ("케이스", "phone case", "휴대폰케이스", "핸드폰케이스"): "3926.90",
        ("자동차", "승용차", "car"): "8703.23",
        ("맥주", "beer"): "2203.00",
        ("쇠고기", "beef", "소고기"): "0201.10",
        ("코트", "coat", "재킷"): "6101.20",
        ("화장품", "cosmetic", "크림"): "3304.99",
    }

    matched_code = None
    for keywords, code in keyword_map.items():
        if any(kw in name_lower for kw in keywords):
            matched_code = code
            break

    if matched_code and matched_code in hs_db:
        entry = hs_db[matched_code]
        return {
            "hs_code": matched_code,
            "description": entry["description"],
            "confidence": 0.92,
            "note": "Mock 분류 결과입니다. 실제 신고 시 관세사 확인 필요.",
        }

    return {
        "hs_code": "UNKNOWN",
        "description": f"'{product_name}'에 대한 HS Code를 찾을 수 없습니다.",
        "confidence": 0.0,
        "note": "관세사 또는 관세청 콜센터(125)에 문의하세요.",
    }


@tool
def get_tariff_rate(hs_code: str, origin_country: str = "일반") -> dict:
    """
    HS Code와 원산지 기준으로 관세율을 조회합니다.

    Args:
        hs_code: 조회할 HS Code (예: "8471.30")
        origin_country: 원산지 국가 (예: "미국", "중국", "일본")

    Returns:
        basic_rate, fta_rate, fta_agreement, special_taxes 포함 dict
    """
    hs_db = _TARIFF_DATA.get("hs_codes", {})
    entry = hs_db.get(hs_code)

    if not entry:
        return {
            "hs_code": hs_code,
            "error": "해당 HS Code 관세율 정보를 찾을 수 없습니다.",
        }

    rates = entry.get("rates", {})
    basic_rate = rates.get("basic", "조회 불가")

    # FTA 세율 조회
    fta_map = {
        "미국": ("fta_us", "한-미 FTA"),
        "us": ("fta_us", "한-미 FTA"),
        "유럽": ("fta_eu", "한-EU FTA"),
        "eu": ("fta_eu", "한-EU FTA"),
        "중국": ("fta_cn", "한-중 FTA"),
        "일본": (None, None),  # 한-일 FTA 미체결
    }
    country_lower = origin_country.lower()
    fta_key, fta_name = None, None
    for country, (key, name) in fta_map.items():
        if country in country_lower:
            fta_key, fta_name = key, name
            break

    fta_rate = rates.get(fta_key) if fta_key else None

    return {
        "hs_code": hs_code,
        "description": entry["description"],
        "origin_country": origin_country,
        "basic_rate": basic_rate,
        "fta_rate": fta_rate,
        "fta_agreement": fta_name if fta_rate else None,
        "special_taxes": entry.get("special_taxes", []),
        "note": entry.get("note", ""),
    }


@tool
def check_import_restrictions(hs_code: str) -> dict:
    """
    HS Code 기준으로 수입 제한/금지 여부를 조회합니다.

    Args:
        hs_code: 조회할 HS Code

    Returns:
        status (ALLOWED/RESTRICTED/PROHIBITED), required_documents 포함 dict
    """
    reg_data = _REGULATION_DATA.get("import_restrictions", {})

    for code_prefix, info in reg_data.items():
        if hs_code.startswith(code_prefix):
            return {
                "hs_code": hs_code,
                "status": info["status"],
                "reason": info.get("reason", ""),
                "required_documents": info.get("required_documents", []),
                "authority": info.get("authority", "관세청"),
            }

    return {
        "hs_code": hs_code,
        "status": "ALLOWED",
        "reason": "특별한 수입 제한 없음",
        "required_documents": ["수입신고서", "상업송장", "포장명세서"],
        "authority": "관세청",
    }


@tool
def get_duty_free_allowance(travel_type: str = "일반") -> dict:
    """
    여행자 면세 한도를 조회합니다.

    Args:
        travel_type: 여행 유형 (일반, 술, 담배 등)
    """
    allowances = _REGULATION_DATA.get("duty_free_allowance", {})
    return allowances.get(
        travel_type,
        allowances.get("일반", {"amount": "$800", "note": "개인 사용 목적"}),
    )
