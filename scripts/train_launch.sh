#!/usr/bin/env bash
# safety-one managed-SFT launcher: Nemotron Nano 3 30B-A3B (headline) and
# Muse Glimmer 30B (hedge, free LoRA SFT via UI/Serverless promo until 2026-11-30).
#
# DRY-RUN by default: prints every command, calls nothing. Pass --execute to run.
#
# Recipe (both jobs): LoRA rank 8, lora-alpha 32, 1 epoch, all 19,830 rows of
# data/nvidia/filtered/train_sft.jsonl, trainer-default learning rate (no
# Nano-specific LR is published; managed SFT picks a per-model default),
# evaluation dataset = dev_sft.jsonl (2,479 rows converted from dev_calib.jsonl).
#
# Upload-blocker note: firectl dataset create / getUploadEndpoint currently
# returns FailedPrecondition "dataset uploads are temporarily disabled".
# When a bucket is available, pass --external-url gs://<bucket>/<file>.jsonl
# (and --eval-external-url) to register the dataset without the upload endpoint;
# the bucket must grant read access to the Fireworks service account
# fireworks-control-plane@fw-ai-cp-prod.iam.gserviceaccount.com (GCS), or use an
# s3:// URI / Azure blob URL per the dataset API. Schemes allowed: gs://, s3://,
# https://<account>.blob.core.windows.net/.
set -euo pipefail

EXECUTE=0
SKIP_NANO=0
SKIP_GLIMMER=0
TRAIN_FILE="data/nvidia/filtered/train_sft.jsonl"
EVAL_FILE="data/nvidia/filtered/dev_sft.jsonl"
DATASET_ID="safety-one-train-sft-v1"
EVAL_DATASET_ID="safety-one-dev-sft-v1"
EXTERNAL_URL=""
EVAL_EXTERNAL_URL=""
EPOCHS=1
LORA_RANK=8
LORA_ALPHA=32

usage() {
  cat <<'USAGE'
Usage: scripts/train_launch.sh [--execute] [options]

  --execute               Run the commands (default: dry-run, prints only).
  --skip-nano             Do not launch the Nano job.
  --skip-glimmer          Do not launch the Glimmer job.
  --train-file PATH       Default: data/nvidia/filtered/train_sft.jsonl
  --eval-file PATH        Default: data/nvidia/filtered/dev_sft.jsonl
  --dataset-id ID         Default: safety-one-train-sft-v1
  --eval-dataset-id ID    Default: safety-one-dev-sft-v1
  --external-url URI      Register train dataset from gs://, s3://, or Azure
                          blob URI instead of uploading (works around the
                          disabled upload endpoint; bucket must be readable by
                          fireworks-control-plane@fw-ai-cp-prod.iam.gserviceaccount.com).
  --eval-external-url URI Same, for the eval dataset.
  -h, --help              This help.

Run from the repo root with firectl authenticated and
FIREWORKS_API_KEY / FIREWORKS_ACCOUNT_ID set (or .env sourced).

Trainer job id / wall-clock / training cost for the shipped adapter: not recorded.
This managed-SFT path is unused by the submission; see scripts/train_api_sft.py.
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --execute) EXECUTE=1 ;;
    --skip-nano) SKIP_NANO=1 ;;
    --skip-glimmer) SKIP_GLIMMER=1 ;;
    --train-file) TRAIN_FILE="$2"; shift ;;
    --eval-file) EVAL_FILE="$2"; shift ;;
    --dataset-id) DATASET_ID="$2"; shift ;;
    --eval-dataset-id) EVAL_DATASET_ID="$2"; shift ;;
    --external-url) EXTERNAL_URL="$2"; shift ;;
    --eval-external-url) EVAL_EXTERNAL_URL="$2"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

run() {
  # Echo the command; execute it only with --execute.
  echo "+ $*"
  if [ "$EXECUTE" -eq 1 ]; then "$@"; fi
}

dataset_ready() {
  firectl dataset get "$1" 2>/dev/null | grep -q "State: READY"
}

ensure_dataset() {
  # $1 dataset id, $2 local file, $3 external url (may be empty)
  local id="$1" file="$2" url="$3"
  if [ "$EXECUTE" -eq 1 ] && dataset_ready "$id"; then
    echo "# dataset $id already READY; skipping upload"
    return 0
  fi
  if [ -n "$url" ]; then
    run firectl dataset create --external-url "$url" --display-name "$id" "$id"
  else
    run firectl dataset create "$id" "$file"
  fi
}

if [ "$EXECUTE" -eq 0 ]; then
  echo "=== DRY RUN (pass --execute to run) ==="
else
  : "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY (source safety-one/.env)}"
  : "${FIREWORKS_ACCOUNT_ID:?set FIREWORKS_ACCOUNT_ID}"
fi

for f in "$TRAIN_FILE" "$EVAL_FILE"; do
  if [ ! -f "$f" ]; then echo "missing $f" >&2; exit 2; fi
done

echo "# Step 1: datasets (train + eval)"
echo "# NOTE: if create fails with FailedPrecondition 'uploads temporarily disabled',"
echo "# re-run with --external-url/--eval-external-url once a readable bucket exists."
echo "# A stale dataset '$DATASET_ID' stuck in UPLOADING can be removed with:"
echo "#   firectl dataset delete $DATASET_ID"
ensure_dataset "$DATASET_ID" "$TRAIN_FILE" "$EXTERNAL_URL"
ensure_dataset "$EVAL_DATASET_ID" "$EVAL_FILE" "$EVAL_EXTERNAL_URL"

echo "# Step 2: launch both LoRA SFT jobs in parallel"
NANO_JOB="safety-one-nano-v1-$(date +%H%M)"
GLIMMER_JOB="safety-one-glimmer-v1-$(date +%H%M)"

if [ "$SKIP_NANO" -eq 0 ]; then
  run firectl sftj create \
    --job-id "$NANO_JOB" \
    --display-name "safety-one-nano-v1" \
    --base-model accounts/fireworks/models/nemotron-nano-3-30b-a3b \
    --dataset "$DATASET_ID" \
    --evaluation-dataset "$EVAL_DATASET_ID" \
    --output-model safety-one-nano-safety \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --epochs "$EPOCHS" &
fi
if [ "$SKIP_GLIMMER" -eq 0 ]; then
  run firectl sftj create \
    --job-id "$GLIMMER_JOB" \
    --display-name "safety-one-glimmer-v1" \
    --base-model accounts/fireworks/models/muse-glimmer-30b \
    --dataset "$DATASET_ID" \
    --evaluation-dataset "$EVAL_DATASET_ID" \
    --output-model safety-one-glimmer-safety \
    --lora-rank "$LORA_RANK" --lora-alpha "$LORA_ALPHA" \
    --epochs "$EPOCHS" &
fi
wait

ACCT="${FIREWORKS_ACCOUNT_ID:-<account-id>}"
cat <<EOF

# Step 3: poll (repeat until SUCCEEDED; --watch blocks)
python scripts/train_poll.py --job-id $NANO_JOB --execute --watch
python scripts/train_poll.py --job-id $GLIMMER_JOB --execute --watch

# Step 4: check deploy shape, then deploy each tuned model (bills GPU-time)
firectl deployment-shape-version match --model accounts/$ACCT/models/safety-one-nano-safety
python scripts/deploy_create.py --model accounts/$ACCT/models/safety-one-nano-safety --auto-stop-minutes 30 --execute
python scripts/deploy_create.py --model accounts/$ACCT/models/safety-one-glimmer-safety --auto-stop-minutes 30 --execute

# Upload-blocked fallback that needs no dataset upload at all (data streams
# from the client): Serverless/Dedicated Training API.
#   Glimmer chat SFT, FREE until 2026-11-30 (serverless pool):
#     pip install -e /tmp/fireworks-cookbook/training
#     python -m examples.sft.muse_glimmer_starter prepare --file $TRAIN_FILE --run-id safety-glimmer-v1
#     python -m examples.sft.muse_glimmer_starter launch --surface serverless --file $TRAIN_FILE --run-id safety-glimmer-v1
#   Nano chat SFT, dedicated shape (bills ~2x B200 GPU-time, unverified price):
#     sft_loop Config(trainer=TrainerConfig(training_shape_id=
#       "accounts/fireworks/trainingShapes/nemotron-nano-3-30b-a3b-262k-b200-lora"))
EOF
