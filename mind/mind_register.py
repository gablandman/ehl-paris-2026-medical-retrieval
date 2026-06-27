from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai>=1.5.0",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "scipy>=1.11",
#   "tqdm>=4.67",
# ]
# ///

"""
Affine pre-registration for dataset2, layered on top of the MIND pipeline.

Why this file exists
--------------------
``mind_retrieval.py`` describes each voxel by the local self-similarity of its
patch (the MIND descriptor) and scores a (query, target) pair by the mean
absolute difference of the two descriptor fields. That mean-|.| is a *spatial*
comparison: it only makes sense when the two volumes already sit in the same
frame. dataset1 (shared grid) and dataset3 (intra-op resampled into the query
space) satisfy that, and MIND is near-perfect there. dataset2 does NOT: it
applies an INDEPENDENT rigid+elastic deformation to the query and the target,
so voxel x in the query no longer corresponds to voxel x in the target and the
descriptor difference becomes mostly noise (offline synthetic-ds2 MRR ~0.44,
top-1 conflict ~0.78).

The fix here is to *re-register before comparing*, but only for dataset2. For a
(query, target) pair we estimate an affine transform that warps the target into
the query frame, then compute the usual MIND dissimilarity on the aligned pair.
Affine cannot undo the elastic part, but the deformation has a large rigid
component, and restoring the coarse frame is enough to make the descriptor
difference informative again.

The registration metric is MIND distance itself. The two modalities differ
(T1 vs T2), so an intensity metric like NCC/SSD is unreliable; MIND is already
our modality-invariant currency, so we minimise the same quantity we will rank
on. The transform is a 12-parameter affine (a 3x4 ``theta`` for
``F.affine_grid``), optimised by Adam on a downsampled grid for speed, with a
small multi-start over the identity plus a few axis flips/rotations to escape
the obvious local minima of brain symmetry.

Two registration strategies are exposed (see ``--strategy``):

  * ``pairwise`` (Option B): register each target to each query independently,
    then MIND. Galleries are <=100, so a ds2 pool is <=100x100 registrations;
    each is cheap on a downsampled grid, so this stays feasible. This is the
    accurate option and the one validated offline.

  * ``template`` (Option A): register every volume once to a single reference
    (the first query), bringing the whole pool into one coarse frame, then MIND
    pairwise as usual. N registrations instead of N^2, but it assumes one global
    frame fits every pair, which the independent per-volume deformation breaks.

Everything dataset1/dataset3 is left untouched: this module only ever rewrites
the two ds2 pools' similarity matrices.

Entry points mirror ``mind_retrieval.py``:

  * ``--mode offline``  -- on the synthetic-ds2 proxy (held-out ds1 pairs with
    an independent rigid+elastic deform on the target), report raw-MIND MRR vs
    registration+MIND MRR. This is the gate: only worth submitting if
    registration clearly lifts the synthetic-ds2 number above the ~0.44 raw
    baseline (and the 0.801 LB it implies).

  * ``--mode submit``  -- compute MIND for ds1/ds3 pools exactly as
    ``mind_retrieval.py`` does, but compute registration+MIND for the ds2 pools,
    then Sinkhorn-rerank and write one submission CSV.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from slice_clip_baseline import (
    collect_prediction_sets,
    load_training_pairs,
)
from rerank_baseline import (
    mrr,
    rank_argmax,
    rank_sinkhorn,
    top1_conflict_rate,
    write_submission,
)
from mind_retrieval import (
    _loader,
    deform_volume,
    load_volume,
    mind_descriptor,
    mind_fields,
    mind_sim_matrix,
)


# --------------------------------------------------------------------------- #
# Gradient-based affine registration (metric = MIND distance)
# --------------------------------------------------------------------------- #


def _theta_init(device: torch.device) -> list[torch.Tensor]:
    """A small bank of affine init matrices (3x4) for multi-start registration.

    Brain volumes are roughly left-right symmetric and the ds2 rigid component
    can be large, so a pure-identity start sometimes lands in a wrong basin.
    We seed with the identity plus a few 180-degree axis rotations; the best
    final loss wins. Translations start at zero (the volumes are resized into a
    common cube, so they are already coarsely centred).
    """
    eye = torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], device=device)
    flips = [
        torch.diag(torch.tensor([1.0, -1.0, -1.0], device=device)),   # 180 about x
        torch.diag(torch.tensor([-1.0, 1.0, -1.0], device=device)),   # 180 about y
        torch.diag(torch.tensor([-1.0, -1.0, 1.0], device=device)),   # 180 about z
    ]
    inits = [eye.clone()]
    for R in flips:
        t = torch.zeros(3, 4, device=device)
        t[:, :3] = R
        inits.append(t)
    return inits


def _mind_at(vol: torch.Tensor, patch_radius: int) -> torch.Tensor:
    """MIND descriptor of a (G,G,G) volume as a (C,G,G,G) field (thin wrapper)."""
    return mind_descriptor(vol, patch_radius=patch_radius)


def register_affine(
    fixed: torch.Tensor,
    moving: torch.Tensor,
    *,
    reg_grid: int = 32,
    iters: int = 150,
    lr: float = 0.02,
    patch_radius: int = 1,
    multi_start: bool = True,
) -> torch.Tensor:
    """Estimate an affine ``theta`` (3x4) that warps ``moving`` onto ``fixed``.

    Both inputs are (G,G,G) intensity volumes in [0,1]. We optimise the affine
    on a downsampled ``reg_grid^3`` copy (registration does not need full
    resolution and this is what makes Option B affordable), minimising the mean
    absolute difference of the MIND descriptor fields of fixed and warped-moving.

    Returns the 3x4 ``theta`` in normalized coordinates, usable directly with
    ``F.affine_grid`` at any resolution.
    """
    device = fixed.device
    g = reg_grid

    def down(v):
        return F.interpolate(
            v.view(1, 1, *v.shape), size=(g, g, g), mode="trilinear", align_corners=False
        )[0, 0]

    f_small = down(fixed)
    m_small = down(moving)
    f_mind = _mind_at(f_small, patch_radius).detach()  # (C,g,g,g)
    size = (1, 1, g, g, g)

    inits = _theta_init(device) if multi_start else [
        torch.tensor([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0]], device=device)
    ]

    best_theta = None
    best_loss = float("inf")
    m_in = m_small.view(1, 1, g, g, g)

    for init in inits:
        theta = init.clone().unsqueeze(0).requires_grad_(True)  # (1,3,4)
        opt = torch.optim.Adam([theta], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            grid = F.affine_grid(theta, size, align_corners=False)
            warped = F.grid_sample(
                m_in, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0, 0]
            w_mind = _mind_at(warped, patch_radius)
            loss = (w_mind - f_mind).abs().mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            grid = F.affine_grid(theta, size, align_corners=False)
            warped = F.grid_sample(
                m_in, grid, mode="bilinear", padding_mode="zeros", align_corners=False
            )[0, 0]
            final = (_mind_at(warped, patch_radius) - f_mind).abs().mean().item()
        if final < best_loss:
            best_loss = final
            best_theta = theta.detach().clone()

    return best_theta  # (1,3,4)


def warp_volume(vol: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Apply a 3x4 affine ``theta`` to a (G,G,G) volume at its native resolution."""
    g = vol.shape
    grid = F.affine_grid(theta, (1, 1, *g), align_corners=False)
    out = F.grid_sample(
        vol.view(1, 1, *g), grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )
    return out[0, 0]


# --------------------------------------------------------------------------- #
# Registration-aware similarity matrices
# --------------------------------------------------------------------------- #


def mind_sim_matrix_pairwise_reg(
    q_fields, q_vols, t_vols, *, patch_radius=1, reg_grid=32, iters=150, lr=0.02,
    multi_start=True, verbose=False,
):
    """Option B: register each target to each query, then MIND. Returns (S, qids, tids).

    For every (query, target) pair we estimate target->query affine on a
    downsampled grid, warp the (full-grid) target, recompute its MIND field, and
    score with the same mean-|.| dissimilarity as the aligned path. Higher = more
    similar.
    """
    qids = sorted(q_fields)
    tids = sorted(t_vols)
    S = np.zeros((len(qids), len(tids)), dtype=np.float32)
    for i, q in enumerate(qids):
        qf = q_fields[q].reshape(-1)
        qv = q_vols[q]
        for j, t in enumerate(tids):
            theta = register_affine(
                qv, t_vols[t], reg_grid=reg_grid, iters=iters, lr=lr,
                patch_radius=patch_radius, multi_start=multi_start,
            )
            warped = warp_volume(t_vols[t], theta)
            tf = mind_descriptor(warped, patch_radius=patch_radius).reshape(-1)
            S[i, j] = -float((qf - tf).abs().mean().item())
        if verbose:
            print(f"    query {i+1}/{len(qids)} registered against {len(tids)} targets", flush=True)
    return S, qids, tids


def mind_sim_matrix_template_reg(
    q_fields, q_vols, t_vols, *, patch_radius=1, reg_grid=32, iters=150, lr=0.02,
    multi_start=True, verbose=False,
):
    """Option A: register every volume to one reference, then MIND. Returns (S, qids, tids).

    Reference = the first query (sorted). All queries and targets are warped into
    that frame once (N registrations), their MIND fields recomputed, then scored
    pairwise as usual.
    """
    qids = sorted(q_fields)
    tids = sorted(t_vols)
    ref = q_vols[qids[0]]

    q_warp_fields = {}
    for k, q in enumerate(qids):
        if q == qids[0]:
            q_warp_fields[q] = q_fields[q]
            continue
        theta = register_affine(ref, q_vols[q], reg_grid=reg_grid, iters=iters, lr=lr,
                                patch_radius=patch_radius, multi_start=multi_start)
        q_warp_fields[q] = mind_descriptor(warp_volume(q_vols[q], theta), patch_radius=patch_radius)
    t_warp_fields = {}
    for k, t in enumerate(tids):
        theta = register_affine(ref, t_vols[t], reg_grid=reg_grid, iters=iters, lr=lr,
                                patch_radius=patch_radius, multi_start=multi_start)
        t_warp_fields[t] = mind_descriptor(warp_volume(t_vols[t], theta), patch_radius=patch_radius)
        if verbose:
            print(f"    target {k+1}/{len(tids)} registered to reference", flush=True)

    S = np.zeros((len(qids), len(tids)), dtype=np.float32)
    T = torch.stack([t_warp_fields[t].reshape(-1) for t in tids])
    for i, q in enumerate(qids):
        qf = q_warp_fields[q].reshape(-1)
        d = (T - qf.unsqueeze(0)).abs().mean(dim=1)
        S[i] = -d.detach().cpu().numpy().astype(np.float32)
    return S, qids, tids


# --------------------------------------------------------------------------- #
# Offline validation on the synthetic-ds2 proxy
# --------------------------------------------------------------------------- #


def run_offline(args, device):
    transform = _loader()
    data_root = args.data_root.resolve()
    pairs = load_training_pairs(data_root, args.train_pair_csv, [], [], [])
    rng = np.random.default_rng(args.seed)

    idx = np.arange(len(pairs))
    rng.shuffle(idx)
    holdout = [pairs[i] for i in idx[: args.holdout]]
    print(f"Offline proxy: {len(holdout)} held-out ds1 pairs (of {len(pairs)}).", flush=True)

    q_paths = {str(p["query_id"]): Path(p["query_path"]) for p in holdout}
    t_paths = {str(p["target_id"]): Path(p["target_path"]) for p in holdout}
    gold = {str(p["query_id"]): str(p["target_id"]) for p in holdout}

    print("Computing MIND fields for held-out queries/targets...", flush=True)
    q_fields, q_vols = mind_fields(q_paths, transform, args.grid, device, args.patch_radius)
    t_fields, t_vols = mind_fields(t_paths, transform, args.grid, device, args.patch_radius)

    print("Building synthetic-ds2 targets (independent rigid+elastic deform)...", flush=True)
    t_vols_def, t_fields_def = {}, {}
    for t, v in t_vols.items():
        vd = deform_volume(v, rng, args.rigid_deg, args.elastic_mag, args.elastic_sigma)
        t_vols_def[t] = vd
        t_fields_def[t] = mind_descriptor(vd, patch_radius=args.patch_radius)

    results = {}

    # raw MIND (no registration) -- the ~0.44 baseline we must beat
    S_raw, qids, tids = mind_sim_matrix(q_fields, q_vols, t_fields_def, t_vols_def, align=False)
    results["synds2_MIND_raw"] = mrr(rank_argmax(S_raw, qids, tids), gold)
    results["synds2_MIND_raw_conflict"] = top1_conflict_rate(S_raw)
    print(f"  raw MIND MRR={results['synds2_MIND_raw']:.4f}", flush=True)

    if args.strategy in ("pairwise", "both"):
        print("Registration (pairwise, Option B)...", flush=True)
        S_reg, _, _ = mind_sim_matrix_pairwise_reg(
            q_fields, q_vols, t_vols_def, patch_radius=args.patch_radius,
            reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
            multi_start=not args.no_multi_start, verbose=True,
        )
        results["synds2_MIND_reg_pairwise"] = mrr(rank_argmax(S_reg, qids, tids), gold)
        results["synds2_MIND_reg_pairwise_conflict"] = top1_conflict_rate(S_reg)
        print(f"  reg(pairwise)+MIND MRR={results['synds2_MIND_reg_pairwise']:.4f}", flush=True)

    if args.strategy in ("template", "both"):
        print("Registration (template, Option A)...", flush=True)
        S_tpl, _, _ = mind_sim_matrix_template_reg(
            q_fields, q_vols, t_vols_def, patch_radius=args.patch_radius,
            reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
            multi_start=not args.no_multi_start, verbose=True,
        )
        results["synds2_MIND_reg_template"] = mrr(rank_argmax(S_tpl, qids, tids), gold)
        results["synds2_MIND_reg_template_conflict"] = top1_conflict_rate(S_tpl)
        print(f"  reg(template)+MIND MRR={results['synds2_MIND_reg_template']:.4f}", flush=True)

    print("\n=== OFFLINE MRR (synthetic-ds2) ===")
    print(json.dumps(results, indent=2))
    return results


# --------------------------------------------------------------------------- #
# Submission: ds1/ds3 unchanged, ds2 registered
# --------------------------------------------------------------------------- #


def run_submit(args, device):
    transform = _loader()
    data_root = args.data_root.resolve()
    prediction_sets = collect_prediction_sets(data_root, args.query_csv, args.gallery_csv)
    reg_pools = set(args.reg_pool)

    all_rows = []
    for idx, ps in enumerate(prediction_sets):
        do_reg = idx in reg_pools
        print(f"\nPool {idx}: Q={len(ps['queries'])} G={len(ps['targets'])} register={do_reg}", flush=True)
        q_fields, q_vols = mind_fields(ps["queries"], transform, args.grid, device, args.patch_radius)
        t_fields, t_vols = mind_fields(ps["targets"], transform, args.grid, device, args.patch_radius)

        if do_reg:
            if args.strategy == "template":
                S, qids, tids = mind_sim_matrix_template_reg(
                    q_fields, q_vols, t_vols, patch_radius=args.patch_radius,
                    reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
                    multi_start=not args.no_multi_start, verbose=True,
                )
            else:
                S, qids, tids = mind_sim_matrix_pairwise_reg(
                    q_fields, q_vols, t_vols, patch_radius=args.patch_radius,
                    reg_grid=args.reg_grid, iters=args.reg_iters, lr=args.reg_lr,
                    multi_start=not args.no_multi_start, verbose=True,
                )
        else:
            # ds1/ds3: identical to mind_retrieval.py (no registration, no align).
            S, qids, tids = mind_sim_matrix(q_fields, q_vols, t_fields, t_vols, align=False)
        print(f"  top-1 conflict rate: {top1_conflict_rate(S):.3f}", flush=True)

        if args.rerank == "sinkhorn":
            all_rows.extend(rank_sinkhorn(S, qids, tids, args.sinkhorn_tau, args.sinkhorn_iter))
        else:
            all_rows.extend(rank_argmax(S, qids, tids))

    write_submission(args.out, all_rows)
    print(f"\nWrote {len(all_rows)} rows -> {args.out}", flush=True)


# --------------------------------------------------------------------------- #


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["offline", "submit"], required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--grid", type=int, default=64, help="common cube side for MIND")
    p.add_argument("--patch-radius", type=int, default=1, help="MIND patch radius (box)")
    p.add_argument("--seed", type=int, default=20260626)

    # registration
    p.add_argument("--strategy", choices=["pairwise", "template", "both"], default="pairwise")
    p.add_argument("--reg-grid", type=int, default=32, help="downsampled grid for registration")
    p.add_argument("--reg-iters", type=int, default=150, help="Adam iters per registration")
    p.add_argument("--reg-lr", type=float, default=0.02, help="Adam lr for the affine params")
    p.add_argument("--no-multi-start", action="store_true",
                   help="disable the flip/rotation multi-start (identity-only init)")

    # offline
    p.add_argument("--train-pair-csv", type=Path, action="append", default=[])
    p.add_argument("--holdout", type=int, default=80)
    p.add_argument("--rigid-deg", type=float, default=10.0)
    p.add_argument("--elastic-mag", type=float, default=0.03)
    p.add_argument("--elastic-sigma", type=float, default=4.0)

    # submit
    p.add_argument("--query-csv", type=Path, action="append", default=[])
    p.add_argument("--gallery-csv", type=Path, action="append", default=[])
    p.add_argument("--reg-pool", type=int, nargs="*", default=[2, 3],
                   help="pool indices to register (ds2 val/test); ds1/ds3 left as-is")
    p.add_argument("--rerank", choices=["argmax", "sinkhorn"], default="sinkhorn")
    p.add_argument("--sinkhorn-tau", type=float, default=10.0)
    p.add_argument("--sinkhorn-iter", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("mind_reg_submission.csv"))
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; grid={args.grid}; reg_grid={args.reg_grid}; "
          f"strategy={args.strategy}; iters={args.reg_iters}", flush=True)
    if args.mode == "offline":
        run_offline(args, device)
    else:
        run_submit(args, device)


if __name__ == "__main__":
    main()
