#!/usr/bin/env bash
# vision_deploy.sh — create the Nemotron 3.5 Super VL on-demand deployment
# (the image route) with a 30-minute auto-stop teardown.
#
# DRY RUN BY DEFAULT: prints the firectl commands and exits. Pass --execute
# to actually create the deployment (coordinator only — billed GPU time).
#
# Cost facts (verified 2026-11-02 via `firectl deployment-shape-version match`
# and https://fireworks.ai/pricing):
#   model : accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-nvfp4
#   shape : accounts/fireworks/deploymentShapes/f3m8sxu2 — 4x NVIDIA_B200_180GB, FP4
#   rate  : 4 GPUs x $13/GPU-h = $52/h, billed per GPU-second while hot
#           (~$0.87/min; 30 min ~= $26). The Fireworks meter is authoritative.
#
# Usage:
#   bash scripts/vision_deploy.sh                 # print commands, no calls
#   bash scripts/vision_deploy.sh --execute       # create + watchdog + wait for READY
#   bash scripts/vision_deploy.sh --execute --minutes 20 --no-wait
set -euo pipefail

MODEL="accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-nvfp4"
SHAPE="accounts/fireworks/deploymentShapes/f3m8sxu2"   # 4x NVIDIA_B200_180GB
GPUS=4
RATE_H=52          # $/h fleet rate, https://fireworks.ai/pricing
MINUTES=30
READY_TIMEOUT_MIN=20
DEPLOYMENT_ID=""
EXECUTE=0
WAIT_READY=1

while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --minutes) MINUTES="$2"; shift ;;
    --ready-timeout-min) READY_TIMEOUT_MIN="$2"; shift ;;
    --model) MODEL="$2"; shift ;;
    --shape) SHAPE="$2"; shift ;;
    --deployment-id) DEPLOYMENT_ID="$2"; shift ;;
    --no-wait) WAIT_READY=0 ;;
    -h|--help) sed -n '1,30p' "$0"; exit 0 ;;
    *) echo "unknown flag: $1" >&2; exit 2 ;;
  esac
  shift
done

if [ -z "$DEPLOYMENT_ID" ]; then
  DEPLOYMENT_ID="safety-one-vl-$(date -u +%m%d%H%M)"
fi
WORKDIR=/tmp/so-vision-deploy
COST_NOTE="${GPUS}x B200 x \$13/GPU-h = \$${RATE_H}/h (~\$0.87/min; ${MINUTES} min cap ~= \$$(( RATE_H * MINUTES / 60 )))"

echo "== Super VL deployment plan (model: $MODEL) =="
echo "Cost while hot: $COST_NOTE  [https://fireworks.ai/pricing]"
echo
echo "# 0. read-only shape check (\$0):"
echo "firectl deployment-shape-version match --model $MODEL"
echo
echo "# 1. create on-demand deployment (billing starts when replicas go hot):"
echo "firectl deployment create $MODEL \\"
echo "    --deployment-id $DEPLOYMENT_ID \\"
echo "    --deployment-shape $SHAPE \\"
echo "    --min-replica-count 1 --max-replica-count 1 \\"
echo "    --display-name safety-one-supervl-eval"
echo
echo "# 2. point the vision route at it:"
echo "export FIREWORKS_VISION_MODEL=accounts/\$FIREWORKS_ACCOUNT_ID/deployments/$DEPLOYMENT_ID"
echo
echo "# 3. auto-stop watchdog (${MINUTES} min hard cap):"
echo "( sleep $((MINUTES * 60)) && firectl deployment delete $DEPLOYMENT_ID --ignore-checks ) &"
echo
echo "# manual teardown any time (--ignore-checks: without it the API refuses to"
echo "# delete a deployment that served requests in the last hour):"
echo "firectl deployment delete $DEPLOYMENT_ID --ignore-checks"

if [ "$EXECUTE" -ne 1 ]; then
  echo
  echo "Dry run — nothing was created. Re-run with --execute to deploy."
  exit 0
fi

if ! command -v firectl >/dev/null 2>&1; then
  echo "firectl not found on PATH" >&2; exit 2
fi
mkdir -p "$WORKDIR"

echo
echo "== shape match (read-only) =="
firectl deployment-shape-version match --model "$MODEL" || \
  echo "WARNING: shape match failed; create may still pick --deployment-shape" >&2

echo
echo "== creating deployment $DEPLOYMENT_ID =="
firectl deployment create "$MODEL" \
  --deployment-id "$DEPLOYMENT_ID" \
  --deployment-shape "$SHAPE" \
  --min-replica-count 1 --max-replica-count 1 \
  --display-name safety-one-supervl-eval | tee "$WORKDIR/create.out"

RESOURCE=$(grep -oE 'accounts/[^[:space:]]+/deployments/[A-Za-z0-9_-]+' "$WORKDIR/create.out" | head -1 || true)
if [ -z "$RESOURCE" ] && [ -n "${FIREWORKS_ACCOUNT_ID:-}" ]; then
  RESOURCE="accounts/${FIREWORKS_ACCOUNT_ID}/deployments/${DEPLOYMENT_ID}"
fi
if [ -z "$RESOURCE" ]; then
  echo "WARNING: could not resolve the deployment resource path; check 'firectl deployment get $DEPLOYMENT_ID'" >&2
  RESOURCE="$DEPLOYMENT_ID"
fi

cat > "$WORKDIR/deployment.env" <<EOF
export FIREWORKS_VISION_MODEL=$RESOURCE
EOF
echo
echo "Deployment resource: $RESOURCE"
echo "export FIREWORKS_VISION_MODEL=$RESOURCE"
echo "(also written to $WORKDIR/deployment.env — source it before eval_vision.py)"

# Hard-cap watchdog: deletes the deployment ${MINUTES} after create returned.
(
  sleep "$((MINUTES * 60))"
  firectl deployment delete "$DEPLOYMENT_ID" --ignore-checks >>"$WORKDIR/teardown.log" 2>&1 || true
) </dev/null >>"$WORKDIR/teardown.log" 2>&1 &
echo "Auto-stop watchdog scheduled: 'firectl deployment delete $DEPLOYMENT_ID --ignore-checks' in ${MINUTES} min (log: $WORKDIR/teardown.log)"

if [ "$WAIT_READY" -eq 1 ]; then
  echo "== waiting up to ${READY_TIMEOUT_MIN} min for READY =="
  deadline=$(( $(date +%s) + READY_TIMEOUT_MIN * 60 ))
  serving=0
  while [ "$(date +%s)" -lt "$deadline" ]; do
    # Top-level "State: Ready" line (firectl prints mixed case); nested shape
    # fields are indented so ^State: only matches the deployment itself.
    state=$(firectl deployment get "$DEPLOYMENT_ID" 2>/dev/null | sed -n 's/^State: *//p' | head -1 | tr '[:lower:]' '[:upper:]' || true)
    echo "  state: ${state:-unknown}  ($(date -u +%H:%M:%S)Z)"
    case "$state" in
      READY|PARTIAL_READY) echo "Deployment serving — run scripts/eval_vision.py now."; serving=1; break ;;
      FAILED|DELETED) echo "Deployment entered $state; see 'firectl deployment events $DEPLOYMENT_ID'" >&2; exit 4 ;;
    esac
    sleep 30
  done
  if [ "$serving" -ne 1 ]; then
    echo "Not READY within ${READY_TIMEOUT_MIN} min — deleting $DEPLOYMENT_ID" >&2
    firectl deployment delete "$DEPLOYMENT_ID" --ignore-checks || true
    exit 5
  fi
fi
echo
echo "Done. Remember: billed per GPU-second while hot; teardown watchdog fires in ${MINUTES} min or run 'firectl deployment delete $DEPLOYMENT_ID --ignore-checks' now."
