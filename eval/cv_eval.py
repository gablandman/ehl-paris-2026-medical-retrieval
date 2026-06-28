from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai>=1.5.0", "torch>=2.7.0", "numpy>=2.0",
#   "nibabel>=5.3", "scipy>=1.11", "tqdm>=4.67",
# ]
# ///

"""
Cross-validation / robustness harness for the MIND retrieval recipes.

Goal
----
Estimate how the two top recipes -- {MIND-only, MIND+ds2-registration} -- behave
under each reranker {argmax, sinkhorn, hungarian}, with mean +/- std, on three
held-out proxies built from the 350 labelled dataset1 pairs:

  * ds1-real   : repeated random holdout over the real aligned pairs (clean).
                 MIND-only and MIND+reg are evaluated; on aligned data reg is a
                 (costly) near-no-op and we want to confirm it does not HURT.
  * synth-ds2  : each held-out target gets an INDEPENDENT rigid+elastic deform.
                 We use a HARDER deformation than the previous proxy (which
                 saturated at MRR=1.0 while real ds2 still had ~0.10-0.14
                 conflict) so the proxy better matches real ds2 difficulty.
  * synth-ds3  : APPROXIMATE pre-op->intra-op proxy: random region erasure
                 (cuboid masking, mimicking resection / missing tissue) plus a
                 mild deformation on the held-out target. Clearly approximate.

For every (recipe, reranker, proxy) we report MRR mean +/- std across repeats
(and seeds), with a bootstrap 95% CI on the pooled per-query reciprocal ranks.
We also report the mean top-1 conflict rate per cell.

This reuses the EXACT MIND + registration code paths from mind_retrieval.py and
mind_register.py, so the numbers are comparable to the submission pipeline.

Run (preferred): ``scripts/run_cv_eval.sh`` (sets PYTHONPATH to core/mind/experiments).
Or directly, from the repo root:
  PYTHONPATH=core:mind:experiments python eval/cv_eval.py --data-root data \
      --train-pair-csv data/dataset1/train_pairs.csv \
      --pool-size 40 --synth-pool-size 24 \
      --ds1-repeats 40 --synth-seeds 5 --reg-repeats 3 \
      --reg-grid 32 --reg-iters 100 --out runs/cv_results.json

Registration defaults match the 0.98796 submission (reg-grid 32, reg-iters 100,
identity init -- multi-start OFF unless --reg-multi-start).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from rerank_baseline import (
    mrr,
    rank_argmax,
    rank_hungarian,
    rank_sinkhorn,
    top1_conflict_rate,
)
from slice_clip_baseline import load_training_pairs
from mind_retrieval import (
    _loader,
    deform_volume,
    load_volume,
    mind_descriptor,
    mind_sim_matrix,
)
from mind_register import mind_sim_matrix_pairwise_reg


RERANKERS = ("argmax", "sinkhorn", "hungarian")


# --------------------------------------------------------------------------- #
# per-query reciprocal ranks (so we can pool + bootstrap, not just average MRR)
# --------------------------------------------------------------------------- #

def reciprocal_ranks(rows, gold):
    rr = []
    for row in rows:
        if row["query_id"] not in gold:
            continue
        ranking = row["target_id_ranking"].split()
        try:
            rr.append(1.0 / (ranking.index(gold[row["query_id"]]) + 1))
        except ValueError:
            rr.append(0.0)
    return rr


def rank_all(S, qids, tids, tau, n_iter):
    return {
        "argmax": rank_argmax(S, qids, tids),
        "sinkhorn": rank_sinkhorn(S, qids, tids, tau, n_iter),
        "hungarian": rank_hungarian(S, qids, tids),
    }


# --------------------------------------------------------------------------- #
# synthetic-ds3 corruption: cuboid region erasure + mild deform
# --------------------------------------------------------------------------- #

def erase_regions(vol, rng, n_regions, max_frac):
    """Zero out n_regions random axis-aligned cuboids (resection proxy).

    max_frac is the max side length of each cuboid as a fraction of the grid.
    Mimics missing tissue: structures present pre-op are gone intra-op.
    """
    g = vol.shape[0]
    out = vol.clone()
    for _ in range(n_regions):
        sz = [int(rng.uniform(0.15, max_frac) * g) for _ in range(3)]
        sz = [max(2, s) for s in sz]
        st = [int(rng.integers(0, max(1, g - s))) for s in sz]
        out[st[0]:st[0]+sz[0], st[1]:st[1]+sz[1], st[2]:st[2]+sz[2]] = 0.0
    return out


# --------------------------------------------------------------------------- #


def load_fields(pairs_subset, transform, grid, device, patch_radius):
    q_paths = {str(p["query_id"]): Path(p["query_path"]) for p in pairs_subset}
    t_paths = {str(p["target_id"]): Path(p["target_path"]) for p in pairs_subset}
    gold = {str(p["query_id"]): str(p["target_id"]) for p in pairs_subset}
    qf, qv = {}, {}
    for k, pth in q_paths.items():
        v = load_volume(pth, transform, grid, device)
        qv[k] = v
        qf[k] = mind_descriptor(v, patch_radius=patch_radius)
    tf, tv = {}, {}
    for k, pth in t_paths.items():
        v = load_volume(pth, transform, grid, device)
        tv[k] = v
        tf[k] = mind_descriptor(v, patch_radius=patch_radius)
    return qf, qv, tf, tv, gold


def summarize(rr_by_repeat):
    """rr_by_repeat: list of per-repeat MRR floats. Return mean/std/CI dict."""
    a = np.array(rr_by_repeat, dtype=np.float64)
    mean = float(a.mean())
    std = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    # bootstrap 95% CI over the repeat-level MRRs
    rng = np.random.default_rng(12345)
    boots = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(2000)] if len(a) > 1 else [mean]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean": round(mean, 4), "std": round(std, 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)], "n": len(a)}


# --------------------------------------------------------------------------- #


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = _loader()
    data_root = args.data_root.resolve()
    pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    print(f"Loaded {len(pairs)} labelled ds1 pairs. device={device} "
          f"grid={args.grid} pool_size={args.pool_size} synth_pool_size={args.synth_pool_size} "
          f"reg_multi_start={args.reg_multi_start}", flush=True)

    results: dict = {"config": vars(args).copy()}
    results["config"]["data_root"] = str(results["config"]["data_root"])
    results["config"]["train_pair_csv"] = [str(p) for p in results["config"]["train_pair_csv"]]

    # ----------------------------------------------------------------- #
    # 1) ds1-real : repeated random holdout, MIND-only (argmax/sink/hung).
    #    Also MIND+reg on a few repeats to confirm reg does not hurt aligned.
    # ----------------------------------------------------------------- #
    print("\n=== ds1-real (repeated random holdout, MIND-only) ===", flush=True)
    rng = np.random.default_rng(args.seed)
    cells_mindonly = {m: [] for m in RERANKERS}
    conf_mindonly = []
    t0 = time.time()
    for rep in range(args.ds1_repeats):
        idx = rng.permutation(len(pairs))[: args.pool_size]
        sub = [pairs[i] for i in idx]
        qf, qv, tf, tv, gold = load_fields(sub, transform, args.grid, device, args.patch_radius)
        S, qids, tids = mind_sim_matrix(qf, qv, tf, tv, align=False)
        conf_mindonly.append(top1_conflict_rate(S))
        ranks = rank_all(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter)
        for m in RERANKERS:
            cells_mindonly[m].append(mrr(ranks[m], gold))
    print(f"  MIND-only done ({args.ds1_repeats} repeats, {time.time()-t0:.0f}s)", flush=True)

    results["ds1_real"] = {
        "MIND-only": {m: summarize(cells_mindonly[m]) for m in RERANKERS},
        "mean_conflict": round(float(np.mean(conf_mindonly)), 4),
    }

    # MIND+reg on a few repeats (slow; aligned -> should ~match MIND-only)
    if args.reg_repeats > 0:
        print("  MIND+reg on aligned ds1 (sanity: must not hurt)...", flush=True)
        rng2 = np.random.default_rng(args.seed + 1)
        reg_cells = {m: [] for m in RERANKERS}
        reg_conf = []
        for rep in range(args.reg_repeats):
            idx = rng2.permutation(len(pairs))[: args.synth_pool_size]
            sub = [pairs[i] for i in idx]
            qf, qv, tf, tv, gold = load_fields(sub, transform, args.grid, device, args.patch_radius)
            Sr, qids, tids = mind_sim_matrix_pairwise_reg(
                qf, qv, tv, patch_radius=args.patch_radius,
                reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
                multi_start=args.reg_multi_start, verbose=False,
            )
            reg_conf.append(top1_conflict_rate(Sr))
            ranks = rank_all(Sr, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter)
            for m in RERANKERS:
                reg_cells[m].append(mrr(ranks[m], gold))
            print(f"    reg repeat {rep+1}/{args.reg_repeats}", flush=True)
        results["ds1_real"]["MIND+reg"] = {m: summarize(reg_cells[m]) for m in RERANKERS}
        results["ds1_real"]["MIND+reg_mean_conflict"] = round(float(np.mean(reg_conf)), 4)

    # ----------------------------------------------------------------- #
    # 2) synth-ds2 : independent rigid+elastic deform (HARDER), several seeds.
    #    Both recipes: MIND-only(raw) and MIND+reg.
    # ----------------------------------------------------------------- #
    print(f"\n=== synth-ds2 (rigid={args.rigid_deg}deg elastic_mag={args.elastic_mag} "
          f"sigma={args.elastic_sigma}, {args.synth_seeds} seeds) ===", flush=True)
    raw_cells = {m: [] for m in RERANKERS}
    reg_cells = {m: [] for m in RERANKERS}
    raw_conf, reg_conf = [], []
    for s in range(args.synth_seeds):
        srng = np.random.default_rng(args.seed + 100 + s)
        idx = srng.permutation(len(pairs))[: args.synth_pool_size]
        sub = [pairs[i] for i in idx]
        qf, qv, tf, tv, gold = load_fields(sub, transform, args.grid, device, args.patch_radius)
        # deform each target independently
        tvd, tfd = {}, {}
        for t, v in tv.items():
            vd = deform_volume(v, srng, args.rigid_deg, args.elastic_mag, args.elastic_sigma)
            tvd[t] = vd
            tfd[t] = mind_descriptor(vd, patch_radius=args.patch_radius)
        # MIND-only (raw, no registration)
        Sraw, qids, tids = mind_sim_matrix(qf, qv, tfd, tvd, align=False)
        raw_conf.append(top1_conflict_rate(Sraw))
        rr = rank_all(Sraw, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter)
        for m in RERANKERS:
            raw_cells[m].append(mrr(rr[m], gold))
        # MIND+reg
        Sreg, qids, tids = mind_sim_matrix_pairwise_reg(
            qf, qv, tvd, patch_radius=args.patch_radius,
            reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
            multi_start=args.reg_multi_start, verbose=False,
        )
        reg_conf.append(top1_conflict_rate(Sreg))
        rr = rank_all(Sreg, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter)
        for m in RERANKERS:
            reg_cells[m].append(mrr(rr[m], gold))
        print(f"  seed {s+1}/{args.synth_seeds}: raw_conf={raw_conf[-1]:.3f} reg_conf={reg_conf[-1]:.3f}", flush=True)
    results["synth_ds2"] = {
        "MIND-only": {m: summarize(raw_cells[m]) for m in RERANKERS},
        "MIND+reg": {m: summarize(reg_cells[m]) for m in RERANKERS},
        "MIND-only_mean_conflict": round(float(np.mean(raw_conf)), 4),
        "MIND+reg_mean_conflict": round(float(np.mean(reg_conf)), 4),
    }

    # ----------------------------------------------------------------- #
    # 3) synth-ds3 : region erasure + mild deform (APPROXIMATE).
    #    In the real pipeline ds3 is NOT registered, so MIND-only is the
    #    operative recipe; we also report MIND+reg to check it doesn't hurt.
    # ----------------------------------------------------------------- #
    print(f"\n=== synth-ds3 (erase {args.ds3_regions} cuboids up to {args.ds3_maxfrac} of side "
          f"+ mild deform, {args.synth_seeds} seeds) [APPROXIMATE] ===", flush=True)
    raw_cells = {m: [] for m in RERANKERS}
    reg_cells = {m: [] for m in RERANKERS}
    raw_conf, reg_conf = [], []
    for s in range(args.synth_seeds):
        srng = np.random.default_rng(args.seed + 200 + s)
        idx = srng.permutation(len(pairs))[: args.synth_pool_size]
        sub = [pairs[i] for i in idx]
        qf, qv, tf, tv, gold = load_fields(sub, transform, args.grid, device, args.patch_radius)
        tvd, tfd = {}, {}
        for t, v in tv.items():
            # mild deform then erase regions (resection-like missing tissue)
            vd = deform_volume(v, srng, args.ds3_rigid_deg, args.ds3_elastic_mag, args.elastic_sigma)
            vd = erase_regions(vd, srng, args.ds3_regions, args.ds3_maxfrac)
            tvd[t] = vd
            tfd[t] = mind_descriptor(vd, patch_radius=args.patch_radius)
        Sraw, qids, tids = mind_sim_matrix(qf, qv, tfd, tvd, align=False)
        raw_conf.append(top1_conflict_rate(Sraw))
        rr = rank_all(Sraw, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter)
        for m in RERANKERS:
            raw_cells[m].append(mrr(rr[m], gold))
        Sreg, qids, tids = mind_sim_matrix_pairwise_reg(
            qf, qv, tvd, patch_radius=args.patch_radius,
            reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
            multi_start=args.reg_multi_start, verbose=False,
        )
        reg_conf.append(top1_conflict_rate(Sreg))
        rr = rank_all(Sreg, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter)
        for m in RERANKERS:
            reg_cells[m].append(mrr(rr[m], gold))
        print(f"  seed {s+1}/{args.synth_seeds}: raw_conf={raw_conf[-1]:.3f} reg_conf={reg_conf[-1]:.3f}", flush=True)
    results["synth_ds3"] = {
        "MIND-only": {m: summarize(raw_cells[m]) for m in RERANKERS},
        "MIND+reg": {m: summarize(reg_cells[m]) for m in RERANKERS},
        "MIND-only_mean_conflict": round(float(np.mean(raw_conf)), 4),
        "MIND+reg_mean_conflict": round(float(np.mean(reg_conf)), 4),
        "note": "APPROXIMATE proxy: cuboid erasure mimics resection; not real intra-op data.",
    }

    print("\n=== CV RESULTS (JSON) ===")
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"Wrote {args.out}", flush=True)
    return results


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--patch-radius", type=int, default=1)
    p.add_argument("--seed", type=int, default=20260626)
    p.add_argument("--pool-size", type=int, default=40,
                   help="queries/targets per ds1-real MIND-only holdout pool (40 ~ real ds2 val)")
    p.add_argument("--synth-pool-size", type=int, default=30,
                   help="pool size for the (slow) registration cells: ds1-reg + synth-ds2/ds3")
    p.add_argument("--reg-multi-start", action="store_true",
                   help="multi-start registration; OFF by default to match the 0.988 submission (--no-multi-start)")
    p.add_argument("--ds1-repeats", type=int, default=30)
    p.add_argument("--synth-seeds", type=int, default=8)
    p.add_argument("--reg-repeats", type=int, default=6,
                   help="MIND+reg repeats on aligned ds1 (sanity, slow)")
    # registration knobs (match the 0.988 submission: reg-iters 100)
    p.add_argument("--reg-grid", type=int, default=32)
    p.add_argument("--reg-iters", type=int, default=100)
    p.add_argument("--reg-lr", type=float, default=0.02)
    # synth-ds2 deformation (HARDER than the prior rigid=10 mag=0.03 proxy)
    p.add_argument("--rigid-deg", type=float, default=20.0)
    p.add_argument("--elastic-mag", type=float, default=0.06)
    p.add_argument("--elastic-sigma", type=float, default=4.0)
    # synth-ds3 corruption
    p.add_argument("--ds3-regions", type=int, default=3)
    p.add_argument("--ds3-maxfrac", type=float, default=0.4)
    p.add_argument("--ds3-rigid-deg", type=float, default=6.0)
    p.add_argument("--ds3-elastic-mag", type=float, default=0.02)
    # rerank
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    p.add_argument("--out", type=str, default="cv_results.json")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
