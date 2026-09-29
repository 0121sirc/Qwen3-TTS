#!/usr/bin/env bash
# Control the Qwen3-TTS Gradio WebUI (qwen_tts.cli.demo, entry point qwen-tts-demo).
#
# Usage: webui.sh [start|stop|status] [--host HOST] [--port PORT] [--model_dir DIR] [--force]
#   start   Launch the WebUI in the background (default when no command is given)
#   stop    Gracefully stop it (TERM -> wait -> KILL)
#   status  Report whether it is running and whether the endpoint responds
#
# NOTE the demo always binds 0.0.0.0 (--ip); --host only selects the address used
# to probe health (default: tailscale IP, else 127.0.0.1).
#
# The 6GB GPU cannot hold this and the OpenAI API server at the same time, and the
# neighbouring CosyVoice project binds the same 8091/8000 ports, so start refuses
# while either of them has an instance up (override: --force).
#
# Model choice: the base checkpoint clones any voice you upload as reference audio
# in the UI; pretrained_models/Qwen3-TTS-12Hz-0.6B-CustomVoice switches it to the
# 9 preset speakers instead (no reference-audio cloning on that checkpoint).
set -euo pipefail

cd "$(dirname "$0")"
HERE="$(pwd)"

# PATH (ffmpeg/ffprobe for gradio) + glibc malloc tuning
source "$HERE/env.sh"

# Default probe host: the tailscale IP, else fall back to 127.0.0.1 with a warning.
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
PORT="${PORT:-8000}"
# base = voice clone from a reference audio uploaded in the demo UI;
# use pretrained_models/Qwen3-TTS-12Hz-0.6B-CustomVoice for the preset speakers.
MODEL_DIR="${MODEL_DIR:-pretrained_models/Qwen3-TTS-12Hz-0.6B-Base}"
# 16 parallel generations would blow the KV cache out of a 6GB card: serialize them.
CONCURRENCY="${CONCURRENCY:-1}"
FORCE="${FORCE:-0}"
PYTHON="$HERE/.conda_env/bin/python"
DEMO_BIN="$HERE/.conda_env/bin/qwen-tts-demo"
RUN_DIR="$HERE/.conda_env/.run"
PID_FILE="$RUN_DIR/webui.pid"
LOG_FILE="$RUN_DIR/webui.log"
API_PID_FILE="$RUN_DIR/openai_api_server.pid"
API_PORT="${API_PORT:-8091}"

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

  start    Start the Qwen3-TTS WebUI in the background (default).
           Waits until the HTTP endpoint is ready (model loading takes ~10s).
  stop     Gracefully stop the WebUI (TERM -> wait -> KILL fallback).
  status   Show whether the WebUI is running and its endpoint responds.

Options:
  --host HOST      Probe host (default: $HOST); the server itself binds 0.0.0.0
  --port PORT      Port (default: $PORT)
  --model_dir DIR  Model directory (default: $MODEL_DIR)
  --force          Start even if the OpenAI API server appears to be running
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
  curl -fsS --max-time 3 --noproxy '*' "http://$HEALTH_HOST:$PORT/" >/dev/null 2>&1
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
  if [[ "$port" == "$API_PORT" ]]; then
    pid_file_alive "$API_PID_FILE" && return 1          # our own API server
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
  is_running && return 1                              # our own webui
  if pid_file_alive "$COSY_WEBUI_PID_FILE"; then
    echo "the CosyVoice webui -> stop it with: $COSY_ROOT/webui.sh stop"
    return 0
  fi
  echo "an unknown process listening on port $port"
  return 0
}

check_gpu_exclusive() {
  [[ "$FORCE" == "1" ]] && return 0
  if pid_file_alive "$API_PID_FILE"; then
    echo "ERROR: the Qwen3-TTS OpenAI API server is running (pid file: $API_PID_FILE, port: $API_PORT)." >&2
    echo "       This 6GB GPU cannot hold both servers at once." >&2
    echo "       Stop it first, then retry:" >&2
    echo "           ./openai_api_server.sh stop" >&2
    echo "       or pass --force if you know there is room." >&2
    return 1
  fi
  if pid_file_alive "$COSY_API_PID_FILE" || pid_file_alive "$COSY_WEBUI_PID_FILE"; then
    echo "ERROR: the CosyVoice project ($COSY_ROOT) has a server running; the 6GB GPU" >&2
    echo "       can hold only one TTS server at a time." >&2
    echo "       Stop it first, then retry:" >&2
    echo "           $COSY_ROOT/openai_api_server.sh stop   # and/or: $COSY_ROOT/webui.sh stop" >&2
    echo "       or pass --force if you know there is room." >&2
    return 1
  fi
  local holder
  if holder="$(describe_holder "$API_PORT")"; then
    echo "ERROR: port $API_PORT is already taken by $holder." >&2
    echo "       (pass --force only if you know what is listening there)" >&2
    return 1
  fi
  if holder="$(describe_holder "$PORT")"; then
    echo "ERROR: port $PORT is already taken by $holder." >&2
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
  # check_gpu_exclusive also refuses a $PORT someone else already holds
  # (describe_holder), which is what keeps health_ok() from answering for a
  # process we never started.
  check_gpu_exclusive

  if [[ ! -x "$DEMO_BIN" ]]; then
    echo "ERROR: $DEMO_BIN not found; is the Qwen3-TTS env installed (pip install -e .)?" >&2
    return 1
  fi

  mkdir -p "$RUN_DIR"
  : > "$LOG_FILE"
  echo "Starting Qwen3-TTS WebUI on http://$HOST:$PORT (model: $MODEL_DIR) ..."
  # --no-flash-attn is mandatory on this card: flash-attn cannot build on sm61, and
  # the demo enables it by default, which would make transformers raise on load.
  nohup "$DEMO_BIN" "$MODEL_DIR" --ip 0.0.0.0 --port "$PORT" \
    --dtype bfloat16 --no-flash-attn --concurrency "$CONCURRENCY" \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" > "$PID_FILE"

  # Liveness before health: a pid that never bound the socket must lose even if
  # another process answers on this port.
  for _ in $(seq 1 120); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "ERROR: WebUI exited during startup. Last log lines:" >&2
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

  echo "WARNING: started (pid $pid) but the endpoint is not ready yet; see $LOG_FILE"
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
