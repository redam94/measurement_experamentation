#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Measurement Design Agent — Startup Script
# Usage:
#   ./run.sh           → starts both API and Streamlit (default)
#   ./run.sh api       → starts only the FastAPI backend
#   ./run.sh ui        → starts only the Streamlit frontend
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV=".venv"
API_PORT=8000
UI_PORT=8501

# Activate virtualenv
if [[ -f "$VENV/bin/activate" ]]; then
    source "$VENV/bin/activate"
else
    echo "ERROR: virtual environment not found at $VENV"
    echo "Create it with:  python3 -m venv .venv && .venv/bin/pip3.13 install -e .[dev]"
    exit 1
fi

# Load .env if present
if [[ -f ".env" ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
    echo "WARNING: ANTHROPIC_API_KEY is not set. Copy .env.example → .env and add your key."
fi

MODE="${1:-all}"

start_api() {
    echo "▶  Starting FastAPI backend on http://localhost:$API_PORT ..."
    uvicorn backend.main:app --host 0.0.0.0 --port "$API_PORT" --reload
}

start_ui() {
    echo "▶  Starting Streamlit UI on http://localhost:$UI_PORT ..."
    streamlit run frontend/Home.py --server.port "$UI_PORT"
}

case "$MODE" in
    api)
        start_api
        ;;
    ui)
        start_ui
        ;;
    all)
        # Start API in background, UI in foreground
        uvicorn backend.main:app --host 0.0.0.0 --port "$API_PORT" --reload &
        API_PID=$!
        echo "▶  API running (PID $API_PID) at http://localhost:$API_PORT"
        trap "kill $API_PID 2>/dev/null; echo 'Stopped API.'" EXIT
        start_ui
        ;;
    *)
        echo "Unknown mode: $MODE  (use: api | ui | all)"
        exit 1
        ;;
esac
