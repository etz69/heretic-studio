#!/usr/bin/env bash
set -uo pipefail

CONTAINER_LOG_PATH="${CONTAINER_LOG_PATH:-/workspace/container.log}"
mkdir -p "$(dirname "${CONTAINER_LOG_PATH}")"
touch "${CONTAINER_LOG_PATH}"

# Mirror container stdout/stderr to a file so the studio can render a docker-log-like console.
exec > >(tee -a "${CONTAINER_LOG_PATH}") 2>&1

export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export HF_HOME="${HF_HOME:-/workspace/.hf}"
mkdir -p /workspace /workspace/.hf

if command -v ollama >/dev/null 2>&1; then
  ollama serve &
  sleep 3
fi

cd /app
uvicorn serve:app --host 0.0.0.0 --port 8888 --no-access-log &
UV_PID=$!
sleep 2

set +e
python /app/heretic_driver.py
HERETIC_RC=$?
set -e

if [[ "${HERETIC_RC}" -ne 0 ]]; then
  echo "[entrypoint] heretic_driver exited with ${HERETIC_RC} (sidecar on :8888 stays up)"
fi

wait "${UV_PID}"
