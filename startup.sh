#!/usr/bin/env bash
set -e

APP_FILE="${APP_FILE:-app.py}"
PORT="${PORT:-8501}"
STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

export STREAMLIT_SERVER_PORT="${PORT}"
export STREAMLIT_SERVER_ADDRESS="${STREAMLIT_SERVER_ADDRESS}"

streamlit run "${APP_FILE}" \
  --server.address="${STREAMLIT_SERVER_ADDRESS}" \
  --server.port="${PORT}" \
  --server.headless=true \
  --browser.gatherUsageStats=false