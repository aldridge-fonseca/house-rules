#!/usr/bin/env bash
# Regenerate the judge report + slide charts + LIVE demo cache in one command.
# Offline, $0: reads receipts only, makes no model calls, starts no lingering servers.
#
#   bash scripts/finalize.sh                     # results/nvidia-v1 (submission receipts)
#   RUN=results/nvidia-v1 VISION=results/vision bash scripts/finalize.sh
#   SKIP_TESTS=1 bash scripts/finalize.sh        # skip the pytest step
#   OURS=final bash scripts/finalize.sh          # pin the demo's ours lane
#
# Scripts that still accept results/frozen-v1 or results/vision regenerate an
# earlier internal run whose receipts are not published. Prefer nvidia-v1.
#
# Picks up new systems automatically (e.g. specialist-cal / full-cal receipts):
# the report charts/tables include them. The demo's ours lane defaults to the
# nvidia-v1 headline system name when present.
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${RUN:-results/nvidia-v1}"
VISION="${VISION:-results/vision}"
IMG_DIR="${IMG_DIR:-docs/img}"
PY="${PY:-.venv/bin/python}"
[ -x "$PY" ] || PY=python3

[ -d "$RUN" ] || { echo "run dir not found: $RUN" >&2; exit 1; }
VISION_ARGS=()
if [ -d "$VISION" ]; then VISION_ARGS=(--vision "$VISION"); else echo "note: no vision run at $VISION" >&2; fi

echo "== 1/4 judge report: $RUN -> $RUN/report/ (+ $IMG_DIR/)"
"$PY" scripts/make_report.py "$RUN" "${VISION_ARGS[@]}" --png-dir "$IMG_DIR"

echo "== 2/4 demo cache from LIVE receipts"
OURS_ARGS=()
[ -n "${OURS:-}" ] && OURS_ARGS=(--ours-system "$OURS")
"$PY" demo/record.py --run "$RUN" "${VISION_ARGS[@]}" "${OURS_ARGS[@]}"

echo "== 3/4 demo smoke (replay + live-classify path)"
"$PY" demo/smoke.py 2>&1 | grep -v '"GET ' || true
"$PY" demo/smoke.py 2>&1 | grep -q '^SMOKE OK' || { echo "demo smoke FAILED" >&2; exit 1; }
"$PY" demo/smoke.py --live 2>&1 | grep -q '^SMOKE OK' || { echo "demo smoke --live FAILED" >&2; exit 1; }

if [ "${SKIP_TESTS:-0}" != "1" ]; then
  echo "== 4/4 pytest (report + demo-related)"
  "$PY" -m pytest -q tests/test_make_report.py
else
  echo "== 4/4 pytest skipped (SKIP_TESTS=1)"
fi

echo
echo "report : $RUN/report/index.html"
echo "charts : $(ls "$IMG_DIR"/"$(basename "$RUN")"-*.png 2>/dev/null | tr '\n' ' ' || true)"
echo "demo   : python3 demo/server.py  ->  http://127.0.0.1:8787/  (?run=0 autostarts)"
