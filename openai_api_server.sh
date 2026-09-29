#!/usr/bin/env bash
# Control the Qwen3-TTS OpenAI-compatible TTS server (openai_tts_server.py).
#
# Usage: openai_api_server.sh [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]
#   start   Launch the server in the background (default when no command is given)
#   stop    Gracefully stop it (TERM -> wait -> KILL)
#   status  Report whether it is running and whether /v1/health responds
#
# Endpoints once up:
#   GET  /v1/health  GET /v1/models  GET /v1/audio/voices  (+ alias /v1/voices)
#   POST /v1/audio/speech   (OpenAI TTS shape, voices from ./voices/ + ./voices-ext/)
#
# The 6GB GPU can hold exactly one TTS server, and the CosyVoice project next door
# binds the same 8091/8000 ports on purpose: a client keeps a single endpoint no
# matter which backend is live, and the port probe doubles as the cross-project
# mutex. This script therefore refuses to start while CosyVoice's server or webui
# is up (override: --force).
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# PATH (ffmpeg for mp3/flac/opus) + glibc malloc tuning
source "$HERE/env.sh"

# Default bind host: the tailscale IP, else fall back to 127.0.0.1 with a warning.
default_host() {
  local ip=""
  ip="$(tailscale ip -4 2>/dev/null | head -n1 || true)"
  if [[ -z "$ip" ]]; then
    ip="$(ip -4 -o addr show tailscale0 2>/dev/null | awk '{split($4,a,"/"); print a[1]}' || true)"
  fi
  if [[ -z "$ip" ]]; then
    echo "WARNING: no tailscale IP found (tailscale not running?); falling back to 127.0.0.1" >&2
    echo "127.0.0.1"
  else
    echo "$ip"
  fi
}

HOST="${HOST:-$(default_host)}"
PORT="${PORT:-8091}"
# base = voice clone from voices/ + voices-ext/; switch to
# pretrained_models/Qwen3-TTS-12Hz-0.6B-CustomVoice for the 9 preset speakers.
MODEL_DIR="${MODEL_DIR:-pretrained_models/Qwen3-TTS-12Hz-0.6B-Base}"
FORCE="${FORCE:-0}"
PYTHON="$HERE/.conda_env/bin/python"
RUN_DIR="$HERE/.conda_env/.run"
PID_FILE="$RUN_DIR/openai_api_server.pid"
LOG_FILE="$RUN_DIR/openai_api_server.log"
WEBUI_PID_FILE="$RUN_DIR/webui.pid"
WEBUI_PORT="${WEBUI_PORT:-8000}"

# The neighbouring CosyVoice project (sibling checkout); empty if absent.
COSY_ROOT="${COSY_ROOT:-$(cd "$HERE/../CosyVoice" 2>/dev/null && pwd || true)}"
COSY_API_PID_FILE="${COSY_ROOT:+$COSY_ROOT/.conda_env/.run/openai_api_server.pid}"
COSY_WEBUI_PID_FILE="${COSY_ROOT:+$COSY_ROOT/.conda_env/.run/webui.pid}"

# 0.0.0.0 is a bind address, not a dialable one.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

usage() {
  cat <<EOF
Usage: $(basename "$0") [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]

  start    Start the Qwen3-TTS OpenAI TTS server in the background (default).
           Waits until /v1/health is ready (model loading takes ~10s).
  stop     Gracefully stop the server (TERM -> wait -> KILL fallback).
  status   Show whether the server is running and healthy.

Options:
  --host HOST      Bind host (default: $HOST)
  --port PORT      Bind port (default: $PORT)
  --model_dir DIR  Model directory (default: $MODEL_DIR)
  --force          Start even if the webui / the CosyVoice project appears to be running
  -h, --help       Show this help
EOF
}

pid_file_alive() {
  local file="$1"
  [[ -f "$file" ]] || return 1
  local pid
  pid="$(cat "$file" 2>/dev/null || true)"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

is_running() { pid_file_alive "$PID_FILE"; }

health_ok() {
  curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/v1/health" >/dev/null 2>&1
}

# True when something is listening on a TCP port -- catches a server that was
# started by hand, since these scripts may bind a tailscale IP rather than
# 127.0.0.1 (in which case the HTTP probe above cannot see it).
port_listening() {
  ss -H -ltn 2>/dev/null | awk '{print $4}' | grep -qE ":$1$"
}

# Label whatever already holds a port we are about to bind; returns 1 when free.
describe_holder() {
  local port="$1"
  port_listening "$port" || return 1
  if [[ "$port" == "$PORT" ]]; then
    is_running && return 1                      # our own (stale) instance
    if pid_file_alive "$COSY_API_PID_FILE"; then
      echo "the CosyVoice API server -> stop it with: $COSY_ROOT/openai_api_server.sh stop"
      return 0
    fi
    if curl -fsS --max-time 2 --noproxy '*' "http://127.0.0.1:$port/v1/health" 2>/dev/null \
        | grep -q "CosyVoice"; then
      echo "the CosyVoice API server -> stop it with: $COSY_ROOT/openai_api_server.sh stop"
      return 0
    fi
    echo "an unknown process listening on port $port"
    return 0
  fi
  pid_file_alive "$WEBUI_PID_FILE" && return 1  # our own webui
  if pid_file_alive "$COSY_WEBUI_PID_FILE"; then
    echo "the CosyVoice webui -> stop it with: $COSY_ROOT/webui.sh stop"
    return 0
  fi
  echo "an unknown process listening on port $port"
  return 0
}

# This card holds ~3.5GB for Qwen3-TTS; CosyVoice needs ~4.7GB on top of nothing.
check_gpu_exclusive() {
  [[ "$FORCE" == "1" ]] && return 0
  if pid_file_alive "$WEBUI_PID_FILE"; then
    echo "ERROR: the Qwen3-TTS webui is running (pid file: $WEBUI_PID_FILE, port: $WEBUI_PORT)." >&2
    echo "       This 6GB GPU cannot hold both servers at once." >&2
    echo "       Stop it first, then retry:" >&2
    echo "           ./webui.sh stop" >&2
    echo "       or pass --force if you know there is room." >&2
    return 1
  fi
  if pid_file_alive "$COSY_API_PID_FILE" 2>/dev/null || pid_file_alive "$COSY_WEBUI_PID_FILE" 2>/dev/null; then
    echo "ERROR: the CosyVoice project ($COSY_ROOT) has a server running; the 6GB GPU" >&2
    echo "       can hold only one TTS server at a time." >&2
    echo "       Stop it first, then retry:" >&2
    echo "           $COSY_ROOT/openai_api_server.sh stop   # and/or: $COSY_ROOT/webui.sh stop" >&2
    echo "       or pass --force if you know there is room." >&2
    return 1
  fi
  local holder
  if holder="$(describe_holder "$PORT")"; then
    echo "ERROR: port $PORT is already taken by $holder." >&2
    echo "       (pass --force only if you know what is listening there)" >&2
    return 1
  fi
  if holder="$(describe_holder "$WEBUI_PORT")"; then
    echo "ERROR: port $WEBUI_PORT is already taken by $holder." >&2
    echo "       (pass --force only if you know what is listening there)" >&2
    return 1
  fi
}

cmd_start() {
  if is_running; then
    if health_ok; then
      echo "Already running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
      return 0
    fi
    echo "Stale process $(cat "$PID_FILE") without a healthy endpoint; restarting."
    cmd_stop
  fi

  # .conda_env/ is gitignored (it *is* the conda env), so a fresh clone ships no
  # interpreter at this path. Fail with the recipe instead of letting nohup die
  # quietly in the log.
  if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python interpreter not found at: $PYTHON" >&2
    echo "  The conda env (.conda_env/) is not part of this checkout. Create it, then retry:" >&2
    echo "      conda create -p .conda_env python=3.10 pip" >&2
    echo "      .conda_env/bin/pip install -e ." >&2
    echo "  re-run: ./openai_api_server.sh start" >&2
    return 1
  fi

  # check_gpu_exclusive also refuses a $PORT that someone else already holds
  # (describe_holder), which is what keeps health_ok() from answering for a
  # process we never started.
  check_gpu_exclusive

  mkdir -p "$RUN_DIR"
  : > "$LOG_FILE"
  echo "Starting Qwen3-TTS OpenAI TTS server on http://$HOST:$PORT (model: $MODEL_DIR) ..."
  nohup "$PYTHON" "$HERE/openai_tts_server.py" --host "$HOST" --port "$PORT" \
    --model_dir "$MODEL_DIR" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  # Model loading takes ~10s (2.4GB bf16 weights) -- much faster than CosyVoice.
  # Liveness before health: a pid that never bound the socket must lose even if
  # another process answers on this port.
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: server exited during startup. Last log lines:" >&2
      tail -20 "$LOG_FILE" >&2 || true
      rm -f "$PID_FILE"
      return 1
    fi
    if health_ok; then
      echo "Up (pid $pid). Log: $LOG_FILE"
      return 0
    fi
    sleep 1
  done

  echo "WARNING: started (pid $pid) but /v1/health is not ready yet; see $LOG_FILE"
  return 0
}

cmd_stop() {
  if ! is_running; then
    rm -f "$PID_FILE"
    echo "Not running."
    return 0
  fi

  local pid
  pid="$(cat "$PID_FILE")"
  echo "Stopping pid $pid ..."
  kill -TERM "$pid" 2>/dev/null || true

  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done

  if kill -0 "$pid" 2>/dev/null; then
    echo "  force killing $pid"
    kill -9 "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  echo "Stopped."
}

cmd_status() {
  if is_running; then
    echo "Running (pid $(cat "$PID_FILE")) on http://$HOST:$PORT"
    if health_ok; then
      echo "Health: ok"
      curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/v1/health" 2>/dev/null && echo
    else
      echo "Health: FAILED (endpoint not responding; model still loading? see $LOG_FILE)"
      return 1
    fi
  else
    echo "Not running."
    return 1
  fi
}

CMD="start"
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|status) CMD="$1"; shift ;;
    --host) HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model_dir) MODEL_DIR="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# HEALTH_HOST/PORT may have changed via options.
HEALTH_HOST="$HOST"
[[ "$HEALTH_HOST" == "0.0.0.0" ]] && HEALTH_HOST="127.0.0.1"

case "$CMD" in
  start) cmd_start ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
esac
