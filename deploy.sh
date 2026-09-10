#!/usr/bin/env bash
# MOSS-VL Realtime (sglang-omni) NPU service launcher.
#
# Usage:
#   bash deploy.sh                # start and wait until healthy
#   bash deploy.sh stop           # stop the service
#   bash deploy.sh restart        # stop, then start
#
# Overridable via environment:
#   MODEL_PATH, HOST, PORT, GPU, GPUS (TP deployment, e.g. "0,1,5,6,9,13,14,15"),
#   TP_SIZE, CONTEXT_LENGTH, MEM_FRACTION, MAX_RUNNING_REQUESTS, LOG_FILE
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${REPO_DIR}"
# Deterministic HCCL listening base port avoids EI0020 bind failures from
# stale TIME_WAIT sockets of previous runs.
export HCCL_IF_BASE_PORT="${HCCL_IF_BASE_PORT:-61000}"
# Per-rank device-socket ports (EI0020: default 16666 collides across
# concurrent TP groups / stale sockets).
export HCCL_NPU_SOCKET_PORT_RANGE="${HCCL_NPU_SOCKET_PORT_RANGE:-62000-62200}"

MODEL_PATH="${MODEL_PATH:-/inspire/sj-ssd3/project/pretrain-test/public/workspace/models/MOSS-VL-Realtime-SGLANG}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
GPU="${GPU:-0}"
GPUS="${GPUS:-}"
TP_SIZE="${TP_SIZE:-1}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MEM_FRACTION="${MEM_FRACTION:-0.70}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
LOG_FILE="${LOG_FILE:-/tmp/moss_vl_realtime_server.log}"
PID_FILE="${PID_FILE:-/tmp/moss_vl_realtime_server.pid}"

service_pids() {
    pgrep -f "run_moss_vl_realtime_server.*--port ${PORT}" || true
}

stop_service() {
    local pids
    pids="$(service_pids)"
    if [ -z "$pids" ]; then
        echo "Service on port ${PORT} is not running."
        return 0
    fi
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    for _ in $(seq 1 20); do
        [ -z "$(service_pids)" ] && break
        sleep 1
    done
    pids="$(service_pids)"
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
    fi
    echo "Service stopped."
}

wait_healthy() {
    local ip
    ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    for _ in $(seq 1 60); do
        if curl -sf --max-time 3 "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
            echo "Service healthy."
            echo "  Local:   http://127.0.0.1:${PORT}"
            echo "  External: http://${ip:-<host-ip>}:${PORT}   (WebSocket: ws://${ip:-<host-ip>}:${PORT}/v1/video/realtime)"
            return 0
        fi
        sleep 5
    done
    echo "ERROR: service did not become healthy within 300s; log: ${LOG_FILE}" >&2
    tail -20 "$LOG_FILE" >&2 || true
    return 1
}

start_service() {
    if [ -n "$(service_pids)" ]; then
        echo "Service already running on port ${PORT} (stop first with: $0 stop)."
        exit 1
    fi
    if [ ! -d "$MODEL_PATH" ]; then
        echo "ERROR: MODEL_PATH does not exist: $MODEL_PATH" >&2
        exit 1
    fi

    if [ -n "$GPUS" ]; then
        TP_SIZE="$(echo "$GPUS" | awk -F, '{print NF}')"
    fi
    local -a tp_args=()
    if [ "$TP_SIZE" -gt 1 ]; then
        if [ -z "$GPUS" ]; then
            echo "ERROR: TP_SIZE>1 requires GPUS (comma-separated device ids)" >&2
            exit 1
        fi
        tp_args=(--tp-size "$TP_SIZE" --gpus "$GPUS")
        echo "TP deployment: tp-size=${TP_SIZE} gpus=${GPUS}"
    fi

    nohup python "$REPO_DIR/examples/run_moss_vl_realtime_server.py" \
        --model-path "$MODEL_PATH" \
        --gpu "$GPU" \
        "${tp_args[@]}" \
        --host "$HOST" \
        --port "$PORT" \
        --context-length "$CONTEXT_LENGTH" \
        --mem-fraction-static "$MEM_FRACTION" \
        --max-running-requests "$MAX_RUNNING_REQUESTS" \
        > "$LOG_FILE" 2>&1 &

    echo $! > "$PID_FILE"
    echo "Starting MOSS-VL realtime server (pid $(cat "$PID_FILE"), log ${LOG_FILE}) ..."
    wait_healthy
}

case "${1:-start}" in
    start) start_service ;;
    stop) stop_service ;;
    restart) stop_service; start_service ;;
    # 4-card deployment (2×TP2): two instances inside plane A (0,1) and (5,6),
    # leaving plane B (9,13,14,15 — logical 4-7) entirely free for later use
    # (it can host a full TP=4 instance on its own).
    start4)
        # TP=2 halves the per-card headroom; moss_vl's KV is ~3.4MB/token, so
        # the caller-protocol shape (4 frames per round) runs with an 8K
        # context and a higher static-memory share. Plain assignment (not
        # prefix form) so the values cannot be shadowed.
        PORT=8000; GPUS="0,1"; CONTEXT_LENGTH=8192; MEM_FRACTION=0.80
        MAX_RUNNING_REQUESTS=2; LOG_FILE=/tmp/moss_vl_realtime_server.log
        start_service
        PORT=8001; GPUS="2,3"; CONTEXT_LENGTH=8192; MEM_FRACTION=0.80
        MAX_RUNNING_REQUESTS=2; LOG_FILE=/tmp/moss_vl_realtime_server_b.log
        start_service
        echo "4-card deployment up: instance A :${PORT:-8000} (NPU 0,1), instance B :${PORT_8:-8001} (NPU 5,6). Spare: NPU 9,13,14,15 (logical 4-7)."
        ;;
    stop4)
        PORT="${PORT:-8000}" stop_service
        PORT="${PORT_8:-8001}" stop_service
        ;;
    # 8-card topology: the NPUs form two HCCS planes (0,1,5,6) and
    # (9,13,14,15); cross-plane P2P is unavailable, so the supported shape is
    # two TP=4 instances — one per plane. Container-internal logical indices:
    # 0-3 -> plane A, 4-7 -> plane B.
    start8)
        PORT="${PORT:-8000}" GPUS="0,1,2,3" MAX_RUNNING_REQUESTS=4 \
            LOG_FILE="${LOG_FILE:-/tmp/moss_vl_realtime_server.log}" start_service
        PORT="${PORT_8:-8001}" GPUS="4,5,6,7" MAX_RUNNING_REQUESTS=4 \
            LOG_FILE="/tmp/moss_vl_realtime_server_b.log" start_service
        echo "8-card deployment up: instance A :${PORT:-8000} (plane 0,1,5,6), instance B :${PORT_8:-8001} (plane 9,13,14,15)"
        ;;
    stop8)
        PORT="${PORT:-8000}" stop_service
        PORT="${PORT_8:-8001}" stop_service
        ;;
    *) echo "Usage: $0 {start|stop|restart|start4|stop4|start8|stop8}" >&2; exit 1 ;;
esac
