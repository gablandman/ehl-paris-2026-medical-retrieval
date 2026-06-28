#!/usr/bin/env bash
# Cross-validation / robustness harness for the MIND retrieval recipes.
# Estimates how {MIND-only, MIND+ds2-reg} x {argmax, sinkhorn, hungarian}
# generalize, on three proxies built from the 350 labelled dataset1 pairs:
#   ds1-real (repeated holdout), synth-ds2 (harder indep. deform), synth-ds3
#   (region-erasure / resection proxy, APPROXIMATE). See CV_RESULTS.md.
#
#   ./run_cv_eval.sh            # full CV (writes runs/cv_results.json)
# Registration knobs default to the 0.98796 submission (reg-grid 32, reg-iters
# 100, identity-init / no multi-start). Override via env, e.g. REG_ITERS=60.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs}"; mkdir -p "$OUTDIR"
GRID="${GRID:-64}"; REG_GRID="${REG_GRID:-32}"; REG_ITERS="${REG_ITERS:-100}"

"$PY" "$REPO/eval/cv_eval.py" \
  --data-root "$DATA" \
  --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
  --grid "$GRID" --pool-size "${POOL_SIZE:-40}" --synth-pool-size "${SYNTH_POOL_SIZE:-24}" \
  --ds1-repeats "${DS1_REPEATS:-40}" --synth-seeds "${SYNTH_SEEDS:-5}" --reg-repeats "${REG_REPEATS:-3}" \
  --reg-grid "$REG_GRID" --reg-iters "$REG_ITERS" \
  --rigid-deg "${RIGID_DEG:-20}" --elastic-mag "${ELASTIC_MAG:-0.06}" --elastic-sigma "${ELASTIC_SIGMA:-4}" \
  --ds3-regions "${DS3_REGIONS:-3}" --ds3-maxfrac "${DS3_MAXFRAC:-0.4}" \
  --ds3-rigid-deg "${DS3_RIGID_DEG:-6}" --ds3-elastic-mag "${DS3_ELASTIC_MAG:-0.02}" \
  --out "$OUTDIR/cv_results.json" \
  "$@"
