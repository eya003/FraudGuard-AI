#!/bin/sh
set -eu

# Single public service: Streamlit listens on $PORT; FastAPI stays loopback-only.
export API_BASE_URL="http://127.0.0.1:8000"
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 &
api_pid=$!
cleanup() { kill "$api_pid" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
exec python -m streamlit run frontend/app.py --server.address 0.0.0.0 --server.port "${PORT:-8501}" --server.headless true
