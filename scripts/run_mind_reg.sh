#!/usr/bin/env bash
# MIND + ds2-affine-registration -- our BEST submission (LB 0.98796).
#   ./run_mind_reg.sh offline   # synthetic-ds2: raw-MIND vs registration+MIND MRR
#   ./run_mind_reg.sh submit    # one Sinkhorn-reranked submission (ds2 pools registered)
#
# ds1/ds3 use plain MIND; only the two ds2 pools get affine-registered
# (target->query, metric = MIND distance) before MIND scoring, because ds2
# applies an independent rigid+elastic deformation that breaks the voxel
# correspondence MIND relies on.
# Validated: offline synthetic-ds2 0.52 -> 1.00; submit ds2 conflict 0.83/0.75 -> 0.10/0.14.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs}"; mkdir -p "$OUTDIR"
GRID="${GRID:-64}"; REG_GRID="${REG_GRID:-32}"; REG_ITERS="${REG_ITERS:-100}"
MODE="${1:-offline}"

if [ "$MODE" = "offline" ]; then
  "$PY" "$REPO/mind/mind_register.py" --mode offline \
    --data-root "$DATA" \
    --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
    --grid "$GRID" --holdout 80 \
    --strategy pairwise --reg-grid "$REG_GRID" --reg-iters "$REG_ITERS" --no-multi-start \
    "${@:2}"
else
  "$PY" "$REPO/mind/mind_register.py" --mode submit \
    --data-root "$DATA" --grid "$GRID" \
    --strategy pairwise --reg-grid "$REG_GRID" --reg-iters "$REG_ITERS" --no-multi-start \
    --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
    --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
    --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
    --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
    --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
    --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
    --reg-pool 2 3 \
    --rerank sinkhorn --sinkhorn-tau 10 --sinkhorn-iter 50 \
    --out "$OUTDIR/mind_reg_submission.csv" \
    "${@:2}"
fi
