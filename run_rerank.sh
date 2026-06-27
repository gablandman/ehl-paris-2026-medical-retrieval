#!/usr/bin/env bash
# Reproduce the Sinkhorn / Hungarian reranking submissions.
# Override the data location with: DATA_ROOT=/path/to/data ./run_rerank.sh
set -euo pipefail
DR="${DATA_ROOT:-$HOME/medretrieval/data}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python}"

"$PYTHON" "$HERE/rerank_baseline.py" \
  --data-root "$DR" \
  --train-pair-csv "$DR/dataset1/train_pairs.csv" \
  --query-csv "$DR/dataset1/val_queries.csv"  --gallery-csv "$DR/dataset1/val_gallery.csv" \
  --query-csv "$DR/dataset1/test_queries.csv" --gallery-csv "$DR/dataset1/test_gallery.csv" \
  --query-csv "$DR/dataset2/val_queries.csv"  --gallery-csv "$DR/dataset2/val_gallery.csv" \
  --query-csv "$DR/dataset2/test_queries.csv" --gallery-csv "$DR/dataset2/test_gallery.csv" \
  --query-csv "$DR/dataset3/val_queries.csv"  --gallery-csv "$DR/dataset3/val_gallery.csv" \
  --query-csv "$DR/dataset3/test_queries.csv" --gallery-csv "$DR/dataset3/test_gallery.csv" \
  --out-dir "$HERE/rerank_out" \
  --sinkhorn-tau 10.0 --sinkhorn-iter 50
