from __future__ import annotations

"""Apply Sinkhorn / argmax / Hungarian reranking to embeddings produced by
slice_clip_ds2aug.py (saved as embeddings.npz with keys 'q::<id>' / 't::<id>').

Reuses the rerankers from rerank_baseline.py so the final submission stacks the
same bijection-aware Sinkhorn step that gave the 0.555 baseline.
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from rerank_baseline import (
    cosine_sim,
    rank_argmax,
    rank_hungarian,
    rank_sinkhorn,
    top1_conflict_rate,
    write_submission,
)


def read_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_pool_ids(query_csv: Path, gallery_csv: Path):
    q = [r["query_id"] for r in read_csv(query_csv)]
    t = [r["target_id"] for r in read_csv(gallery_csv)]
    return q, t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--query-csv", type=Path, action="append", required=True)
    p.add_argument("--gallery-csv", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("ds2aug_out"))
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.embeddings)
    q_emb = {k[3:]: data[k] for k in data.files if k.startswith("q::")}
    t_emb = {k[3:]: data[k] for k in data.files if k.startswith("t::")}
    print(f"loaded {len(q_emb)} query / {len(t_emb)} target embeddings")

    methods = {"argmax": [], "sinkhorn": [], "hungarian": []}
    for i, (qcsv, gcsv) in enumerate(zip(args.query_csv, args.gallery_csv)):
        qids, tids = load_pool_ids(qcsv, gcsv)
        S, qids2, tids2 = cosine_sim({q: q_emb[q] for q in qids}, {t: t_emb[t] for t in tids})
        print(f"pool {i}: Q={len(qids2)} G={len(tids2)} conflict={top1_conflict_rate(S):.3f}")
        methods["argmax"].extend(rank_argmax(S, qids2, tids2))
        methods["sinkhorn"].extend(rank_sinkhorn(S, qids2, tids2, args.sinkhorn_tau, args.sinkhorn_iter))
        methods["hungarian"].extend(rank_hungarian(S, qids2, tids2))

    for name, rows in methods.items():
        out = args.out_dir / f"{name}_submission.csv"
        write_submission(out, rows)
        print(f"{name}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
