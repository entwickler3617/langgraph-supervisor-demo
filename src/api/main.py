"""
FastAPI 애플리케이션 진입점

★ 시니어 포인트 #4: Observability
  - structlog으로 구조화 JSON 로그 출력
  - 요청마다 correlation_id 자동 주입
  - /health 엔드포인트로 운영 모니터링 지원
"""
import time
import uuid

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routers.chat import router as chat_router
from src.api.schemas import HealthResponse
from src.config import get_settings

# ── 구조화 로깅 설정 ─────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.processors.JSONRenderer(),
    ]
)

log = structlog.get_logger(__name__)
settings = get_settings()

# ── FastAPI 앱 ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="관세청 AI 상담 시스템",
    description=(
        "LangGraph Supervisor 패턴으로 구현된 관세 멀티에이전트 상담 API.\n\n"
        "HS코드 분류, 관세율 조회, 수출입 규정 검색을 4개 전문 에이전트가 협력하여 처리합니다."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 상관 ID 미들웨어 ────────────────────────────────────────────────────
@app.middleware("http")
async def correlation_id_middleware(request: Request, call_next):
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)

    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000

    log.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        elapsed_ms=round(elapsed, 2),
    )

    response.headers["X-Correlation-ID"] = correlation_id
    structlog.contextvars.clear_contextvars()
    return response


# ── 전역 예외 핸들러 ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("unhandled_exception", path=request.url.path, error=str(exc))
    return JSONResponse(
        status_code=500,
        content={"detail": "서버 내부 오류가 발생했습니다. 잠시 후 다시 시도해주세요."},
    )


# ── 라우터 등록 ──────────────────────────────────────────────────────────────
app.include_router(chat_router)


# ── 헬스체크 ─────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health() -> HealthResponse:
    """운영 헬스체크 엔드포인트."""
    return HealthResponse(
        status="ok",
        version="1.0.0",
        mock_mode=settings.use_mock_llm,
    )


@app.get("/", tags=["ops"])
async def root() -> dict:
    return {
        "service": "관세청 AI 상담 시스템",
        "docs": "/docs",
        "health": "/health",
        "mock_mode": settings.use_mock_llm,
    }
