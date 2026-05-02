#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUT_DIR="${OUT_DIR:-./runs/cddsl_severe_$(date +%Y%m%d_%H%M%S)}"

CLIENTS="${CLIENTS:-50}"
ROUNDS="${ROUNDS:-40}"
SAMPLES_PER_CLIENT="${SAMPLES_PER_CLIENT:-1000}"
CROSS_SAMPLES_PER_CLIENT="${CROSS_SAMPLES_PER_CLIENT:-200}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-6}"
LR="${LR:-0.01}"

CONNECTION_RATE="${CONNECTION_RATE:-0.7}"
MIXING_TEMPERATURE="${MIXING_TEMPERATURE:-0.5}"
MIXING_BLEND="${MIXING_BLEND:-0.5}"
MIXING_MIN_SELF="${MIXING_MIN_SELF:-0.0}"

C0="${C0:-0.05}"
C1="${C1:-0.15}"
C2="${C2:-0.15}"
C0_DECAY="${C0_DECAY:-0.95}"

EVAL_POP_EVERY="${EVAL_POP_EVERY:-5}"
RANDOM_ERASING_PROB="${RANDOM_ERASING_PROB:-0.25}"

HET_COUNTS=(${HET_COUNTS:-20 15 10 5})
HET_ALPHAS=(${HET_ALPHAS:-0.1 0.5 1.0 10.0})

mkdir -p "$OUT_DIR"/{logs,metrics,checkpoints}

PYTHONUNBUFFERED=1 "$PYTHON_BIN" -u CDDSL.py \
  --dataset CIFAR10 \
  --data-root "$DATA_ROOT" \
  --device "$DEVICE" \
  --clients "$CLIENTS" \
  --rounds "$ROUNDS" \
  --samples-per-client "$SAMPLES_PER_CLIENT" \
  --cross-samples-per-client "$CROSS_SAMPLES_PER_CLIENT" \
  --batch-size "$BATCH_SIZE" \
  --local-epochs "$LOCAL_EPOCHS" \
  --lr "$LR" \
  --lr-schedule step \
  --lr-gamma 0.5 \
  --lr-decay-every 10 \
  --topology random_rate \
  --connection-rate "$CONNECTION_RATE" \
  --mixing-temperature "$MIXING_TEMPERATURE" \
  --mixing-blend "$MIXING_BLEND" \
  --mixing-min-self-weight "$MIXING_MIN_SELF" \
  --c0 "$C0" \
  --c1 "$C1" \
  --c2 "$C2" \
  --c0-decay "$C0_DECAY" \
  --eval-population-every "$EVAL_POP_EVERY" \
  --random-erasing-prob "$RANDOM_ERASING_PROB" \
  --num-workers 0 \
  --save-metrics \
  --metrics-dir "$OUT_DIR/metrics" \
  --metrics-prefix cddsl_severe \
  --save-checkpoint \
  --checkpoint-dir "$OUT_DIR/checkpoints" \
  --checkpoint-prefix cddsl_severe \
  --no-download \
  --data-style non_iid \
  --split-type diri_groups \
  --heterogeneity-counts "${HET_COUNTS[@]}" \
  --heterogeneity-alphas "${HET_ALPHAS[@]}" \
  2>&1 | tee "$OUT_DIR/logs/cddsl_severe.log"

echo
echo "CD-DSL severe heterogeneity run finished."
echo "Heterogeneity groups: counts=${HET_COUNTS[*]}  alphas=${HET_ALPHAS[*]}"
echo "Logs:        $OUT_DIR/logs"
echo "Metrics CSV: $OUT_DIR/metrics"
echo "Checkpoints: $OUT_DIR/checkpoints"
echo "$OUT_DIR" > "$OUT_DIR/.completed"
