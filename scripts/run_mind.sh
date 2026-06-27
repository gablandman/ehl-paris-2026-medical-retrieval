#!/usr/bin/env bash
# MIND retrieval (our standalone 0.801 result).
#   ./run_mind.sh offline    # validate on ds1-holdout + synthetic-ds2 proxies
#   ./run_mind.sh submit     # write a single Sinkhorn-reranked submission
# Pool order: 0 ds1val 1 ds1test 2 ds2val 3 ds2test 4 ds3val 5 ds3test
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO/core:$REPO/mind:$REPO/experiments:${PYTHONPATH:-}"
PY="${PY:-python}"
DATA="${DATA_ROOT:-${ROOT:-$HOME/medretrieval}/data}"
OUTDIR="${OUTDIR:-$REPO/runs}"; mkdir -p "$OUTDIR"
GRID="${GRID:-64}"
MODE="${1:-offline}"

if [ "$MODE" = "offline" ]; then
  "$PY" "$REPO/mind/mind_retrieval.py" --mode offline \
    --data-root "$DATA" \
    --train-pair-csv "$DATA/dataset1/train_pairs.csv" \
    --grid "$GRID" --holdout 80 --try-align \
    --cosine-npz "${COSINE_NPZ:-$OUTDIR/similarities.npz}" \
    --blend-weights 0.3 0.5 0.7
else
  "$PY" "$REPO/mind/mind_retrieval.py" --mode submit \
    --data-root "$DATA" --grid "$GRID" \
    --query-csv "$DATA/dataset1/val_queries.csv"  --gallery-csv "$DATA/dataset1/val_gallery.csv" \
    --query-csv "$DATA/dataset1/test_queries.csv" --gallery-csv "$DATA/dataset1/test_gallery.csv" \
    --query-csv "$DATA/dataset2/val_queries.csv"  --gallery-csv "$DATA/dataset2/val_gallery.csv" \
    --query-csv "$DATA/dataset2/test_queries.csv" --gallery-csv "$DATA/dataset2/test_gallery.csv" \
    --query-csv "$DATA/dataset3/val_queries.csv"  --gallery-csv "$DATA/dataset3/val_gallery.csv" \
    --query-csv "$DATA/dataset3/test_queries.csv" --gallery-csv "$DATA/dataset3/test_gallery.csv" \
    --align-pool 2 3 \
    --rerank sinkhorn --sinkhorn-tau 10 --sinkhorn-iter 50 \
    --out "$OUTDIR/mind_submission.csv" \
    "${@:2}"
fi
