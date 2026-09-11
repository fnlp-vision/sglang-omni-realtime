#!/usr/bin/env bash
# Performance / correctness probe for the caller-compatible /v1/realtime
# vision protocol (start → ready → frames → frame_ack → output → stop).
#
# Usage:
#   bash perf.sh                       # 5 rounds × 4 synthetic frames on :8000
#   HOST=10.244.66.107 PORT=8001 bash perf.sh
#   FRAMES_DIR=<dir-with-jpegs> bash perf.sh   # real frames instead of synthetic
#
# Gates (from the caller contract): handshake 10s, ready/ack 10s,
# per-round total 10s. The script exits non-zero when a round fails or the
# p95 total exceeds the limit.
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
ROUNDS="${ROUNDS:-5}"
NFRAMES="${NFRAMES:-4}"
FRAMES_DIR="${FRAMES_DIR:-}"
PROMPT="${PROMPT:-请描述这些画面中正在发生的事情。}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
TOTAL_LIMIT_MS="${TOTAL_LIMIT_MS:-10000}"

URL="ws://${HOST}:${PORT}/v1/realtime"
OUT="$(mktemp /tmp/moss_perf_XXXX.json)"

FRAME_ARGS=()
if [ -n "$FRAMES_DIR" ] && [ -d "$FRAMES_DIR" ]; then
    # One --image per frame when real frames are provided (up to NFRAMES).
    i=0
    for f in $(ls "$FRAMES_DIR" | head -n "$NFRAMES"); do
        FRAME_ARGS+=(--image "$FRAMES_DIR/$f")
    done
    # perf_probe repeats the single --image; for real sequences run it once
    # per image is not needed — pass the first frame and keep it simple.
    FRAME_ARGS=(--image "$FRAMES_DIR/$(ls "$FRAMES_DIR" | head -n 1)")
fi

echo "== perf: url=$URL rounds=$ROUNDS frames=$NFRAMES =="
python "$REPO_DIR/perf_probe.py" \
    --url "$URL" \
    --rounds "$ROUNDS" \
    --frames "$NFRAMES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --prompt "$PROMPT" \
    ${FRAME_ARGS+"${FRAME_ARGS[@]}"} \
    --json | tee "$OUT"

RC=${PIPESTATUS[0]}
python - "$OUT" "$TOTAL_LIMIT_MS" <<'EOF'
import json, sys
data = json.load(open(sys.argv[1]))
limit = float(sys.argv[2])
total = data.get("total_ms") or {}
p95 = total.get("p95") or total.get("max") or 0
failures = data.get("failures", data.get("rounds", 1))
ok = failures == 0 and p95 <= limit
print()
print(f"gate: failures={failures} total_p95={p95}ms limit={limit:.0f}ms -> "
      + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
EOF
GATE=$?
rm -f "$OUT"
exit $(( RC || GATE ))
