# Fireworks × NVIDIA — evidence

Receipt-only page for prize eligibility (Fireworks stack + NVIDIA Best-Use). Every claim cites a path in this repo. Unknowns are marked **not recorded**.

Account id in live model paths: `kldjalkdjwwpadjapdj`.

---

## NVIDIA models (exact Fireworks model ids)

| Role | Fireworks model id | How used | Receipt |
|---|---|---|---|
| Specialist base + untuned ablation | `accounts/fireworks/models/nemotron-nano-3-30b-a3b` | LoRA base; also served untuned on on-demand deployment `o4ubjdmu` | `config/train.yaml` (`model.base_model`); `results/nvidia-v1/OURS_SUMMARY.md`; `results/nvidia-v1/nano-untuned.jsonl` (`prediction.model`) |
| Tuned specialist (LoRA artifact) | `accounts/kldjalkdjwwpadjapdj/models/safety-one-nano-safety` | Live-merge on-demand; eval via deployments `qd2gx85j` (nvidia_test / specialist-only / final-rawtext) and `ier7tjje` (disguised_v1 rows in `final.jsonl`) | `results/nvidia-v1/OURS_SUMMARY.md`; `results/nvidia-v1/final.jsonl`, `specialist-only.jsonl`, `ours_timing.json` |
| Gray-zone escalation + large zero-shot baseline | `accounts/fireworks/models/nemotron-3-ultra-nvfp4` | Fireworks **serverless** chat; `nvidia-ultra-zeroshot` baseline; escalation target in `full` / `final` | `config/systems.yaml` (`nvidia-ultra-zeroshot`); `config/router.yaml` (`providers.escalation`); `results/nvidia-v1/ultra-untuned.jsonl`; escalated rows in `final.jsonl` (`…>accounts/fireworks/models/nemotron-3-ultra-nvfp4`) |
| Vision catalog id (on-demand only) | `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-nvfp4` | Wired for image route; deploy recipe in `scripts/vision_deploy.sh` | `config/router.yaml` (`providers.vision`); `config/systems.yaml` (`vision-routed`); `scripts/vision_deploy.sh` |
| Cheap text baseline (catalog) | `accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b` | Configured as `nvidia-lightning-zeroshot` / `vision-text-only` | `config/systems.yaml` only |
| Hedge base (not in final runs) | `accounts/fireworks/models/muse-glimmer-30b` | Fallback / free-SFT recipe in train scripts | `config/train.yaml` (`fallback` / `fast_fallback`); `scripts/train_api_sft.py` (`--job glimmer`); `scripts/train_launch.sh` |

**Published LIVE text runs did not use Lightning or Glimmer** — no `nemotron-lightning-*` or `safety-one-glimmer-*` model strings appear in `results/nvidia-v1/*.jsonl`. Vision Super VL: deploy/eval scripts exist; **no** `results/` JSONL for a Super VL run is shipped here → deployment id / LIVE vision $ / wall time for Super VL: **not recorded** in published results.

---

## NVIDIA dataset

| Item | Value | Receipt |
|---|---|---|
| Dataset | `nvidia/Nemotron-3.5-Content-Safety-Dataset` (HF; CC-BY-4.0) | `data/nvidia/manifest.json`; `scripts/prep_nvidia.py`; `data/README.md` |
| Prep | `scripts/prep_nvidia.py` → chat-SFT + eval splits; leakage filter → `data/nvidia/filtered/` | `data/nvidia/manifest.json`; `data/LEAKAGE.md` |
| Train file used by Training API recipe | `data/nvidia/filtered/train_sft.jsonl` — **19,830** rows | `data/README.md`; `data/LEAKAGE.md`; `scripts/train_api_sft.py` |
| Train eval / loss split | `data/nvidia/filtered/dev_sft.jsonl` — **2,479** rows (`dev_calib` → chat-SFT) | `data/README.md`; `scripts/train_api_sft.py`; `scripts/train_launch.sh` |
| Held-out headline test | `data/nvidia/test.jsonl` — **3,340** text-safety rows (`nvidia_test`) | `data/README.md`; `results/nvidia-v1/OURS_SUMMARY.md` |

How: text-only JSON-label SFT (`{"label","categories"}`) under the project taxonomy; never train on `validation`/`test` (`config/train.yaml` `exclude_splits`). Eval rows in `results/nvidia-v1/` carry `source: nvidia-nemotron-cs-3.5`.

---

## Fireworks features — used vs prepared

| Feature | Status | Evidence |
|---|---|---|
| **Training API LoRA SFT** (streamed local JSONL → dedicated trainer → promote account model) | **Used** for the specialist | Recipe: `scripts/train_api_sft.py --job nano` (base Nano, shape `accounts/fireworks/trainingShapes/nemotron-nano-3-30b-a3b-262k-b200-lora`, LoRA rank **8** / alpha **32**, 1 epoch, batch 32, lr `1e-4`, `output_model_id=safety-one-nano-safety`). Data path named in `data/README.md` (“training API run”). Promoted model id in `results/nvidia-v1/OURS_SUMMARY.md`. Fireworks trainer/session id: **not recorded** in this repo. Train wall-clock and billed training GPU-$: **not recorded** here (script notes ~2× B200; rate “not separately published”). |
| **Managed LoRA SFT** (`firectl dataset create` + `firectl sftj create`) | **Prepared; not evidenced as completed** | Scripts: `scripts/train_sft.py`, `scripts/train_upload.py`, `scripts/train_launch.sh`, `scripts/train_poll.py`. `train_launch.sh` documents upload endpoint `FailedPrecondition` / “dataset uploads are temporarily disabled”. **No** managed `sftj` job id appears under `results/`. |
| **Dataset upload** (`firectl dataset create <id> <file>`) | **Not evidenced as completed** for the winning run | Same scripts; Training API path streams local files and “never calls the disabled dataset-upload endpoint” (`train_api_sft.py` docstring). Optional `--external-url` register path is documented only. |
| **On-demand deployment** (live-merge LoRA / base Nano) | **Used** | Create: `scripts/deploy_create.py` → `firectl deployment create … --deployment-shape default`. Teardown: `scripts/deploy_teardown.py` → `firectl deployment delete <ID> --ignore-checks`. Live ids in receipts: `qd2gx85j`, `o4ubjdmu`, `ier7tjje` (`OURS_SUMMARY.md`, `ours_timing.json`, JSONLs). Create/delete timestamps for those deployments: **not recorded** in shipped files (only eval wall-seconds in `ours_timing.json`). |
| **On-demand Super VL** (4× B200) | **Scripted; LIVE receipts not shipped** | `scripts/vision_deploy.sh` (shape `accounts/fireworks/deploymentShapes/f3m8sxu2`, $52/h catalog note). No `results/*/vision-*.jsonl` in this tree. |
| **Serverless inference** | **Used** | Ultra baseline + escalation: `config/systems.yaml` prices $0.60 / $2.40 per 1M tokens; measured `cost_usd` on `ultra-untuned.jsonl` / escalated `final.jsonl` rows. |
| **firectl / CLI** | **Used in recipes** | Shape match: `firectl deployment-shape-version match` (`train_launch.sh`, `vision_deploy.sh`). Deploy/delete as above. Dataset/SFT CLI prepared in `train_*.py` / `train_launch.sh`. |
| **Decision API / System One** (`POST /systemone`) | **Not used in published runs** | Client exists (`safety_one/providers/fireworks.py` `SystemOneProvider`; registry key `fireworks_systemone`). No published system in `config/systems.yaml` selects it; no `/systemone` model id in `results/nvidia-v1/*.jsonl`. `train_kway.jsonl` (decision_loop rows) is explicitly unused by the final model (`data/README.md`). |

---

## Reproduce commands

Env: `FIREWORKS_API_KEY`, `FIREWORKS_ACCOUNT_ID` (see `.env.example`). Never commit secrets.

### 1) Data (offline, $0)

```bash
python3 scripts/prep_nvidia.py
python3 scripts/check_leakage.py   # writes filtered/ + leakage report
```

### 2) Training config (Training API path that matches the promoted adapter)

Defaults from `scripts/train_api_sft.py` (overrides `config/train.yaml` alpha-16 draft for managed SFT):

```bash
# Dedicated Nano LoRA → accounts/$FIREWORKS_ACCOUNT_ID/models/safety-one-nano-safety
python3 scripts/train_api_sft.py --job nano \
  --train-path data/nvidia/filtered/train_sft.jsonl \
  --eval-path data/nvidia/filtered/dev_sft.jsonl \
  --epochs 1 --batch-size 32 --learning-rate 1e-4 \
  --lora-rank 8 --lora-alpha 32 --max-seq-len 2048
```

Managed-SFT alternative (dry-run by default; upload may still be blocked):

```bash
bash scripts/train_launch.sh
# or:
python3 scripts/train_sft.py \
  --base-model accounts/fireworks/models/nemotron-nano-3-30b-a3b \
  --dataset-file data/nvidia/filtered/train_sft.jsonl \
  --dataset-id safety-one-train-sft-v1 \
  --output-model safety-one-nano-safety \
  --epochs 1 --lora-rank 8 --max-examples 20000
# add --execute only when uploads work; then:
python3 scripts/train_poll.py --job-id <JOB_ID> --execute --watch
```

### 3) Deploy specialist (bills GPU-time while hot) + teardown

```bash
firectl deployment-shape-version match \
  --model accounts/$FIREWORKS_ACCOUNT_ID/models/safety-one-nano-safety

python3 scripts/deploy_create.py --execute --auto-stop-minutes 30 \
  --model accounts/$FIREWORKS_ACCOUNT_ID/models/safety-one-nano-safety
# prints Deployment ID

export FIREWORKS_MODEL=accounts/$FIREWORKS_ACCOUNT_ID/deployments/<ID>

# hard stop (required after traffic; API refuses delete without the flag):
firectl deployment delete <ID> --ignore-checks
# or: python3 scripts/deploy_teardown.py --deployment-id <ID> --execute
```

Untuned Nano ablation: same deploy flow with base
`accounts/fireworks/models/nemotron-nano-3-30b-a3b` (receipt deployment `o4ubjdmu`).

### 4) Super VL deploy (optional; not required for shipped text results)

```bash
bash scripts/vision_deploy.sh                 # dry-run
bash scripts/vision_deploy.sh --execute --minutes 30
export FIREWORKS_VISION_MODEL=accounts/$FIREWORKS_ACCOUNT_ID/deployments/<ID>
firectl deployment delete <ID> --ignore-checks
```

### 5) Eval systems used in `results/nvidia-v1/`

```bash
# full pipeline = final (tuned deploy + Ultra escalation)
python3 scripts/run_eval.py --run nvidia-v1 --concurrency 48 \
  --system final=scripts.nano_ladder:full \
  --data data/nvidia/test.jsonl

# baselines
python3 scripts/run_eval.py --run nvidia-v1 --concurrency 48 \
  --system ultra-untuned=safety_one.registry:nvidia-ultra-zeroshot \
  --data data/nvidia/test.jsonl

# tables from shipped receipts ($0):
python3 scripts/nvidia_summary.py
```

---

## Time and $ (from shipped receipts)

GPU rate used in allocation math: **$13/h per B200** (`results/nvidia-v1/OURS_SUMMARY.md`, `ours_timing.json` note). Ultra serverless: **$0.60 / $2.40 per 1M** in/out (`config/systems.yaml`).

| Step | Time (receipt) | $ (receipt) |
|---|---|---|
| Training API Nano LoRA | **not recorded** | **not recorded** |
| Managed `sftj` / dataset upload | n/a (no completed job id) | n/a |
| Deploy `qd2gx85j` lifetime (create→delete) | **not recorded** | **not recorded** |
| `final` eval wall on `nvidia_test` | **49.3 s**, 3,340 calls (`ours_timing.json`) | GPU alloc **$0.1781** + Ultra tokens **$0.0096** → **~$56 / 1M msgs** (`final_headline.json`) |
| `specialist-only` on `nvidia_test` | **48.3 s**, 3,340 calls | GPU **~$0.174** alloc at $13/h (`ours_timing.json`); tokens $0 in receipts |
| `nano-untuned` on `nvidia_test` (`o4ubjdmu`) | **59.5 s**, 3,339 calls | GPU alloc **$0.2149** (`final_headline.json`) |
| `nano-untuned` on `disguised_v1` | **38.6 s**, 2,205 calls | GPU alloc **$0.1392** |
| `final-rawtext` on `disguised_v1` | **33.3 s**, 2,216 calls | alloc via `ours_timing.json` |
| Ultra zero-shot `nvidia_test` | p50 **710 ms** (`OURS_SUMMARY.md`) | measured tokens **$0.3603** (`final_headline.json` / `ultra-untuned.jsonl`) |
| Ultra zero-shot `disguised_v1` | p50 **713 ms** | measured tokens **$0.3297** |
| `final` escalations on `nvidia_test` | 107 / 3,340 rows (**3.2%**) | included in final token **$0.0096** |
| Super VL on-demand | **not recorded** (no shipped vision run) | script cites **$52/h** catalog rate only |
| Idle / spin-up outside eval windows | **not recorded** | excluded from $/1M method in `OURS_SUMMARY.md` |

Frontier router spend in `results/spend.json` is **not** Fireworks (OpenAI/Anthropic via token router) and is omitted here.

---

## Explicit non-claims

- No claim that managed `firectl sftj` completed.
- No claim that Fireworks dataset upload succeeded for the final adapter.
- No claim that System One / decision-loop training ran on the submission model.
- No claim that Muse Glimmer or Nemotron Lightning appear in the shipped LIVE text results.
- No claim of a shipped LIVE Super VL score (scripts/config only).
