# Convenience targets. Every target uses uv so no activation is needed.
.PHONY: install test lint mcp api ui demo eval db audit

install:
	uv sync --extra test

test:
	uv run pytest

lint:
	uvx ruff check src tests ui scripts && uvx ruff format --check src tests ui scripts

mcp:
	uv run finagent-mcp

api:
	uv run finagent-api

ui:
	uv run streamlit run ui/app.py

# Start MCP and API in the background, then Streamlit in the foreground.
# Ctrl-C stops all three.
demo:
	@trap 'kill 0' EXIT INT TERM; \
	uv run finagent-mcp & \
	sleep 1; uv run finagent-api & \
	sleep 2; uv run streamlit run ui/app.py

# Requires a running API (make demo, or make mcp + make api).
eval:
	uv run python scripts/eval_matrix.py --provider "$${LLM_PROVIDER:-see .env}"

db:
	uv run python -m finagent.ingestion build --raw-dir data/raw/sec --output data/finagent.db

audit:
	uv run python -m finagent.ingestion audit --require-real-enrichment
