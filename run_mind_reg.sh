#!/usr/bin/env bash
# Helper to run the MIND + ds2-affine-registration experiment on the GPU server.
#
#   ./run_mind_reg.sh offline   # synthetic-ds2 proxy: raw-MIND vs registration+MIND MRR
#   ./run_mind_reg.sh submit    # one Sinkhorn-reranked submission (ds2 pools registered)
#
# Layered on top of mind_retrieval.py. dataset1/dataset3 are scored with plain
# MIND exactly as before; only the two dataset2 pools get affine-registered
# (target->query, metric = MIND distance) before MIND scoring, because ds2
# applies an independent rigid+elastic deformation that breaks the voxel
# correspondence MIND's spatial comparison relies on.
#
# Pool ordering for --query-csv/--gallery-csv below is:
#   0 ds1 val   1 ds1 test   2 ds2 val   3 ds2 test   4 ds3 val   5 ds3 test
# so the ds2 pools (2,3) are the ones passed to --reg-pool in submit mode.
#
# Validated config (identity-init affine, no multi-start, 100 iters, reg-grid 32):
#   offline synthetic-ds2 MRR: 0.52 raw -> 1.00 registered (conflict 0.70 -> 0.00)
#   submit ds2 top-1 conflict: 0.83/0.75 (val/test) -> 0.10/0.14

set -euo pipefail
PY=${PY:-~/venv/bin/python}
ROOT=${ROOT:-~/medretrieval}
DATA=${DATA:-$ROOT/data}
GRID=${GRID:-64}
REG_GRID=${REG_GRID:-32}
REG_ITERS=${REG_ITERS:-100}
MODE=${1:-offline}

cd "$ROOT"

if [ "$MODE" = "offline" ]; then
  $PY mind_register.py --mode offline \
    --data-root "$DATA" \
    --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
    --grid "$GRID" --holdout 80 \
    --strategy pairwise --reg-grid "$REG_GRID" --reg-iters "$REG_ITERS" --no-multi-start \
    "${@:2}"
else
  $PY mind_register.py --mode submit \
    --data-root "$DATA" \
    --grid "$GRID" \
    --strategy pairwise --reg-grid "$REG_GRID" --reg-iters "$REG_ITERS" --no-multi-start \
    --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
    --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
    --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
    --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
    --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
    --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
    --reg-pool 2 3 \
    --rerank sinkhorn --sinkhorn-tau 10 --sinkhorn-iter 50 \
    --out mind_reg_submission.csv \
    "${@:2}"
fi
