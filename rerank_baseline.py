"""Assignment-based rerankers (Sinkhorn, Hungarian) on top of the SliceCLIP baseline.

Retrains the same model as slice_clip_baseline.py (same seed / hyperparams), then
for every (dataset, split) query/gallery pool:
  1. builds the full query x target cosine-similarity matrix,
  2. produces three rankings:
       - argmax    : sort each row by raw similarity (parity with baseline)
       - sinkhorn  : exp(tau*S) -> doubly-stochastic via Sinkhorn -> sort
       - hungarian : linear assignment, assigned target first, then descending sim
  3. writes <method>_submission.csv plus a sim-matrix .npz cache.

Diagnostics:
  - Top-1 conflict rate per pool (how often the same target wins multiple queries
    under argmax). High = strong signal that bijection-aware rerankers will help.
  - Labelled MRR on the training set (biased; near 1.0 expected after 500 epochs
    of fit, so mainly a sanity check that rerankers don't break ties wrongly).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from slice_clip_baseline import (
    CONFIG,
    PairImageDataset,
    SliceCLIP,
    collect_prediction_sets,
    embed_images,
    load_training_pairs,
    make_image_dataset,
    train_model,
)


def cosine_sim(query_emb: dict[str, np.ndarray], target_emb: dict[str, np.ndarray]):
    qids = sorted(query_emb)
    tids = sorted(target_emb)
    Q = np.stack([query_emb[q] for q in qids]).astype(np.float32)
    T = np.stack([target_emb[t] for t in tids]).astype(np.float32)
    return Q @ T.T, qids, tids


def logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=axis, keepdims=True))).squeeze(axis)


def sinkhorn_log(S: np.ndarray, tau: float, n_iter: int) -> np.ndarray:
    """Log-domain Sinkhorn on exp(tau*S). Returns log-post matrix."""
    logK = tau * S
    logu = np.zeros(S.shape[0], dtype=np.float64)
    logv = np.zeros(S.shape[1], dtype=np.float64)
    for _ in range(n_iter):
        logu = -logsumexp(logK + logv[None, :], axis=1)
        logv = -logsumexp(logK + logu[:, None], axis=0)
    return logu[:, None] + logK + logv[None, :]


def rank_argmax(S: np.ndarray, qids, tids):
    return [
        {"query_id": qids[i], "target_id_ranking": " ".join(tids[j] for j in np.argsort(-S[i]))}
        for i in range(len(qids))
    ]


def rank_sinkhorn(S: np.ndarray, qids, tids, tau: float, n_iter: int):
    return rank_argmax(sinkhorn_log(S, tau, n_iter), qids, tids)


def rank_hungarian(S: np.ndarray, qids, tids):
    row_idx, col_idx = linear_sum_assignment(-S)
    assigned = {int(r): int(c) for r, c in zip(row_idx, col_idx)}
    rows = []
    for i in range(len(qids)):
        order = list(np.argsort(-S[i]))
        if i in assigned:
            t_star = assigned[i]
            order = [t_star] + [j for j in order if j != t_star]
        rows.append({"query_id": qids[i], "target_id_ranking": " ".join(tids[j] for j in order)})
    return rows


def mrr(rankings, gold: dict[str, str]) -> float:
    rr = []
    for row in rankings:
        if row["query_id"] not in gold:
            continue
        ranking = row["target_id_ranking"].split()
        try:
            rr.append(1.0 / (ranking.index(gold[row["query_id"]]) + 1))
        except ValueError:
            rr.append(0.0)
    return float(np.mean(rr)) if rr else 0.0


def top1_conflict_rate(S: np.ndarray) -> float:
    """Fraction of queries whose argmax target was also picked by another query."""
    top1 = np.argmax(S, axis=1)
    _, inverse, counts = np.unique(top1, return_inverse=True, return_counts=True)
    return float((counts[inverse] > 1).sum() / len(top1))


def write_submission(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    p.add_argument("--query-csv", type=Path, action="append", required=True)
    p.add_argument("--gallery-csv", type=Path, action="append", required=True)
    p.add_argument("--out-dir", type=Path, default=Path("rerank_out"))
    p.add_argument("--ckpt", type=Path, default=Path("rerank_out/model.pt"))
    p.add_argument("--sim-cache", type=Path, default=Path("rerank_out/similarities.npz"))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt.parent.mkdir(parents=True, exist_ok=True)

    random.seed(int(CONFIG["seed"]))
    np.random.seed(int(CONFIG["seed"]))

    example_root = Path(__file__).resolve().parent
    data_root = args.data_root.resolve()

    train_pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)

    train_images: dict[str, Path] = {}
    for pair in train_pairs:
        train_images[str(pair["query_id"])] = Path(pair["query_path"])
        train_images[str(pair["target_id"])] = Path(pair["target_path"])

    inference_images: dict[str, Path] = {}
    for ps in prediction_sets:
        inference_images.update(ps["queries"])
        inference_images.update(ps["targets"])
    # also embed train images so we can compute labelled-train MRR diagnostics
    inference_images.update(train_images)

    print(json.dumps({
        "config": CONFIG,
        "num_train_images": len(train_images),
        "num_inference_images": len(inference_images),
        "num_train_pairs": len(train_pairs),
        "num_prediction_sets": len(prediction_sets),
    }, indent=2))

    train_image_dataset = make_image_dataset(train_images, example_root, augment=True)
    inference_image_dataset = make_image_dataset(inference_images, example_root, augment=False)
    train_dataset = PairImageDataset(train_pairs, train_image_dataset)

    device = torch.device(str(CONFIG["device"]))
    if args.skip_train and args.ckpt.exists():
        print(f"Loading model from {args.ckpt}")
        model = SliceCLIP(int(CONFIG["embedding_dim"])).to(device)
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    else:
        print("Training model (this matches the existing 0.37659 baseline)...")
        model = train_model(train_dataset)
        torch.save(model.state_dict(), args.ckpt)
        print(f"Saved checkpoint -> {args.ckpt}")

    print("\nEmbedding all images...")
    q_emb_all = embed_images(model, inference_image_dataset, encoder="query")
    t_emb_all = embed_images(model, inference_image_dataset, encoder="target")

    # --- Diagnostic: labelled MRR on training pairs (biased) ---
    train_qids = sorted({str(p["query_id"]) for p in train_pairs})
    train_tids = sorted({str(p["target_id"]) for p in train_pairs})
    gold = {str(p["query_id"]): str(p["target_id"]) for p in train_pairs}
    S_tr, qids_tr, tids_tr = cosine_sim(
        {q: q_emb_all[q] for q in train_qids},
        {t: t_emb_all[t] for t in train_tids},
    )
    print(f"\n=== Labelled-train diagnostic (queries={len(qids_tr)}, gallery={len(tids_tr)}; biased) ===")
    print(f"top-1 conflict rate (argmax): {top1_conflict_rate(S_tr):.3f}")
    print(f"  argmax   MRR={mrr(rank_argmax(S_tr, qids_tr, tids_tr), gold):.4f}")
    print(f"  sinkhorn MRR={mrr(rank_sinkhorn(S_tr, qids_tr, tids_tr, args.sinkhorn_tau, args.sinkhorn_iter), gold):.4f}")
    print(f"  hungari. MRR={mrr(rank_hungarian(S_tr, qids_tr, tids_tr), gold):.4f}")

    # --- Per-pool submissions + conflict diagnostics ---
    methods = {"argmax": [], "sinkhorn": [], "hungarian": []}
    sim_cache: dict[str, np.ndarray] = {}
    print("\n=== Per-pool conflict + ranking ===")
    for idx, ps in enumerate(prediction_sets):
        q_emb = {qid: q_emb_all[qid] for qid in ps["queries"]}
        t_emb = {tid: t_emb_all[tid] for tid in ps["targets"]}
        S, qids, tids = cosine_sim(q_emb, t_emb)
        sim_cache[f"S_{idx}"] = S
        sim_cache[f"qids_{idx}"] = np.array(qids)
        sim_cache[f"tids_{idx}"] = np.array(tids)
        print(f"  pool {idx}: Q={len(qids)} G={len(tids)} top-1 conflict rate={top1_conflict_rate(S):.3f}")
        methods["argmax"].extend(rank_argmax(S, qids, tids))
        methods["sinkhorn"].extend(rank_sinkhorn(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter))
        methods["hungarian"].extend(rank_hungarian(S, qids, tids))

    np.savez(args.sim_cache, **sim_cache)
    print(f"\nSaved similarity cache -> {args.sim_cache}")

    for name, rows in methods.items():
        out = args.out_dir / f"{name}_submission.csv"
        write_submission(out, rows)
        print(f"  {name}: {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
