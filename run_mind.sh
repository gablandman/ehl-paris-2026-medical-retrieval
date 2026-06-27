#!/usr/bin/env bash
# Helper to run the MIND retrieval experiment on the GPU server.
#
#   ./run_mind.sh offline    # validate on ds1-holdout + synthetic-ds2 proxies
#   ./run_mind.sh submit     # write a single Sinkhorn-reranked submission
#
# Pool ordering for --query-csv/--gallery-csv below is:
#   0 ds1 val   1 ds1 test   2 ds2 val   3 ds2 test   4 ds3 val   5 ds3 test
# so ds2 pools (2,3) are the ones to COM-align in submit mode.

set -euo pipefail
PY=${PY:-~/venv/bin/python}
ROOT=${ROOT:-~/medretrieval}
DATA=${DATA:-$ROOT/data}
COSINE_NPZ=${COSINE_NPZ:-$ROOT/rerank_out/similarities.npz}
GRID=${GRID:-64}
MODE=${1:-offline}

cd "$ROOT"

if [ "$MODE" = "offline" ]; then
  $PY mind_retrieval.py --mode offline \
    --data-root "$DATA" \
    --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
    --grid "$GRID" --holdout 80 --try-align \
    --cosine-npz "$COSINE_NPZ" \
    --blend-weights 0.3 0.5 0.7
else
  $PY mind_retrieval.py --mode submit \
    --data-root "$DATA" \
    --grid "$GRID" \
    --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
    --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
    --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
    --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
    --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
    --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
    --align-pool 2 3 \
    --rerank sinkhorn --sinkhorn-tau 10 --sinkhorn-iter 50 \
    --out mind_submission.csv \
    "${@:2}"
fi
