.PHONY: install run ui test lint clean docker-up docker-down

# ── 설치 ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	cp -n .env.example .env || true

# ── 개발 실행 ─────────────────────────────────────────────────────────────────
run:
	@echo "Starting API server (Mock mode)..."
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	@echo "Starting Streamlit UI..."
	streamlit run src/ui/streamlit_app.py --server.port 8501

# ── 테스트 ───────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report: htmlcov/index.html"

# ── 코드 품질 ─────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/

# ── Docker ───────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# ── 정리 ─────────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf data/chroma_db/

# ── 빠른 데모 ─────────────────────────────────────────────────────────────────
demo:
	@echo "=== 관세청 AI 상담 시스템 데모 ==="
	curl -s -X POST http://localhost:8000/api/v1/chat \
	  -H "Content-Type: application/json" \
	  -d '{"session_id":"demo-001","message":"노트북 미국에서 수입 관세율"}' \
	  | python -m json.tool
