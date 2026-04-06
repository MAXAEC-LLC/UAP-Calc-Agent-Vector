#!/usr/bin/env bash
# ─── UAP local dev launcher ────────────────────────────────────────
# Starts backend (uvicorn) and frontend (next dev) as fully detached
# processes that survive terminal closures and VS Code session resets.
#
# Usage:  ./scripts/run-local.sh          # pick random free ports
#         ./scripts/run-local.sh 45123    # fixed backend port, random frontend
#         ./scripts/run-local.sh 45123 35095   # both fixed
#         ./scripts/run-local.sh stop     # kill running servers
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PIDFILE_BE="/tmp/uap-backend.pid"
PIDFILE_FE="/tmp/uap-frontend.pid"
LOGFILE_BE="/tmp/uap-backend.log"
LOGFILE_FE="/tmp/uap-frontend.log"

# ── helpers ─────────────────────────────────────────────────────────
free_port() {
  python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1',0)); print(s.getsockname()[1]); s.close()"
}

kill_if_running() {
  local pidfile="$1" label="$2"
  if [[ -f "$pidfile" ]]; then
    local pid
    pid=$(<"$pidfile")
    if kill -0 "$pid" 2>/dev/null; then
      echo "  stopping $label (PID $pid)"
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pidfile"
  fi
  # also kill anything left on the port
  local port="${3:-}"
  if [[ -n "$port" ]]; then
    local leftover
    leftover=$(lsof -ti:"$port" 2>/dev/null || true)
    if [[ -n "$leftover" ]]; then
      echo "  cleaning port $port (PIDs $leftover)"
      echo "$leftover" | xargs kill -9 2>/dev/null || true
      sleep 0.5
    fi
  fi
}

# ── stop mode ───────────────────────────────────────────────────────
if [[ "${1:-}" == "stop" ]]; then
  echo "Stopping UAP servers…"
  kill_if_running "$PIDFILE_BE" "backend"
  kill_if_running "$PIDFILE_FE" "frontend"
  echo "Done."
  exit 0
fi

# ── load .env if present ────────────────────────────────────────────
if [[ -f "${ROOT}/.env" ]]; then
  set -a
  source "${ROOT}/.env"
  set +a
fi

# ── resolve ports ───────────────────────────────────────────────────
BACKEND_PORT="${1:-$(free_port)}"
FRONTEND_PORT="${2:-$(free_port)}"

# ── stop any previous instances ─────────────────────────────────────
echo "Cleaning up previous instances…"
kill_if_running "$PIDFILE_BE" "backend" "$BACKEND_PORT"
kill_if_running "$PIDFILE_FE" "frontend" "$FRONTEND_PORT"

# ── start backend ───────────────────────────────────────────────────
echo ""
echo "Starting backend on http://127.0.0.1:${BACKEND_PORT} …"

PYTHON="${ROOT}/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="$(command -v python3)"
fi

setsid "$PYTHON" -m uvicorn main:app \
  --host 127.0.0.1 --port "$BACKEND_PORT" --app-dir "${ROOT}/backend" \
  --timeout-keep-alive 120 \
  > "$LOGFILE_BE" 2>&1 &
echo $! > "$PIDFILE_BE"
disown

# ── start frontend ──────────────────────────────────────────────────
echo "Starting frontend on http://127.0.0.1:${FRONTEND_PORT} …"

cd "${ROOT}/web"
NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL:-}" \
BACKEND_INTERNAL_URL="http://127.0.0.1:${BACKEND_PORT}" \
  setsid npm run dev -- --hostname 127.0.0.1 --port "$FRONTEND_PORT" \
  > "$LOGFILE_FE" 2>&1 &
echo $! > "$PIDFILE_FE"
disown

# ── wait for readiness ──────────────────────────────────────────────
echo ""
echo "Waiting for servers…"
for i in $(seq 1 20); do
  be_up=false; fe_up=false
  curl -sS "http://127.0.0.1:${BACKEND_PORT}/api/settings" >/dev/null 2>&1 && be_up=true
  curl -sS "http://127.0.0.1:${FRONTEND_PORT}" >/dev/null 2>&1 && fe_up=true
  if $be_up && $fe_up; then break; fi
  sleep 1
done

echo ""
echo "═══════════════════════════════════════════"
if $be_up; then
  echo "  Backend  ✓  http://127.0.0.1:${BACKEND_PORT}   PID $(cat "$PIDFILE_BE")"
else
  echo "  Backend  ✗  failed — check $LOGFILE_BE"
fi
if $fe_up; then
  echo "  Frontend ✓  http://127.0.0.1:${FRONTEND_PORT}   PID $(cat "$PIDFILE_FE")"
else
  echo "  Frontend ✗  failed — check $LOGFILE_FE"
fi
echo "═══════════════════════════════════════════"
echo ""
echo "Logs:  tail -f $LOGFILE_BE $LOGFILE_FE"
echo "Stop:  $0 stop"
