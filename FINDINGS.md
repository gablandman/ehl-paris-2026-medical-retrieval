# Findings & Improvement Roadmap

Brain MRI Cross-Modal Retrieval Challenge (`ehl-paris-medical-image-retrieval`).
Team **BRAINROT-LABS**. Living document — update as experiments land.

> **Metric:** mean reciprocal rank (MRR) of the true target, averaged over
> dataset1 / dataset2 / dataset3. Only the rank of the *correct* target matters.

---

## 1. Current standing

| Submission | Public MRR | Notes |
|---|---|---|
| SliceCLIP baseline (GPU) | 0.37659 | provided baseline, adapted to run on MI300X |
| SliceCLIP + Sinkhorn rerank | 0.55487 | prior best (learned bi-encoder) |
| **MIND + Sinkhorn (training-free)** | **0.80088** | **best — currently 2nd place** (see §4.6) |

Leaderboard at last check: leader **0.99210**, us **0.80088** (2nd), then 0.561, 0.557, 0.503.

**Headroom: ~0.19 MRR to the leader — and it is ENTIRELY dataset2.** MIND scores ~1.0 on
ds1/ds3 but only ~0.40 on ds2 (its independent deformation breaks voxel correspondence). The
single highest-value lever now is **rigid/affine pre-registration of query↔candidate before
MIND on ds2** (almost certainly what the 0.99 leader did).

**Still our best: SliceCLIP + Sinkhorn (0.555).** Since then we explored a pretrained-encoder
line (BrainIAC) across four experiments — all came in *below* baseline (see §4.4). The lesson
they share now drives strategy: gains must come from **dataset2/3 signal**, not more dataset1
adaptation.

---

## 2. The data (measured, not assumed)

Three **independent** retrieval pools. Always rank a query only against its own
dataset + split gallery; never mix.

| Dataset | Character | train pairs | val (q/gal) | test (q/gal) | Volume shape |
|---|---|---|---|---|---|
| dataset1 | Clean, all registered to one common grid | **350** | 40/40 | 100/100 | 240×240×155 |
| dataset2 | Same source, but val/test get **independent rigid + elastic deformations** | none | 40/40 | 100/100 | 240×240×155 |
| dataset3 | **Pre-op → intra-op** pairs; anatomy genuinely differs | none | 20/20 | 77/77 | **variable** (e.g. 211×250×192, 219×250×192, 256×256×192) |

- Modalities: query = **T1 post-contrast**, target = **T2** for every labelled pair.
- Voxel spacing always **1×1×1 mm**. dataset3 shapes vary → never assume a fixed shape.
- **No intensity normalization** in the release: min is 0, max ranges wildly (~210 to ~13,775 across volumes). Per-volume scaling is essential.
- 1,454 NIfTI volumes total (~17 MB each). On the server the manifests say `.nii.gz`
  but the files are uncompressed `.nii` (handled in code).
- **Only dataset1 has labels (350 pairs).** dataset2/3 are pure generalization tests.

---

## 3. The one big structural fact: it's a one-to-one matching

In every pool `|queries| == |gallery|`. In the ideal answer each gallery scan is the
best match for **exactly one** query. The baseline ignores this and ranks each query
independently, so the same target gets chosen #1 by many queries.

We measured the **top-1 conflict rate** (share of queries whose argmax target is also
some other query's argmax). It is high everywhere — and highest where the domain gap is
largest:

| Pool | top-1 conflict rate |
|---|---|
| ds1 val / test | 0.475 / 0.550 |
| ds2 val / test | **0.775** / 0.710 |
| ds3 val / test | 0.650 / 0.701 |

This is *the* reason bijection-aware reranking (Sinkhorn/Hungarian) works.

---

## 4. Experiments run so far

1. **Baseline on GPU** — adapted device→`cuda` (ROCm) and `.nii`/`.nii.gz` path fallback.
   500 epochs, loss 4.77 → 0.66. → 0.37659.
2. **Bijection reranking** (Gabriel's design) on the baseline embeddings:
   - argmax (no reranking) 0.481, **Sinkhorn 0.555**, Hungarian 0.529 — *same embeddings*.
   - **Sinkhorn (soft) beats Hungarian (hard).** Makes sense: ds2/ds3 are deformed /
     structurally different, so a strict permutation is too aggressive; the soft version
     stays robust. This is our **first confirmed amelioration axis (done).**

### ⚠️ Variance warning (important)
The *same* code path (argmax) scored **0.481** on a fresh model vs the original baseline's
**0.377** — a ~0.10 MRR swing purely from training nondeterminism (GPU non-determinism +
random augmentation order). **Single submissions are noisy.** Control seeds, and prefer
averaging / a local validation signal over reading one LB number.

### 4.4 Pretrained-encoder line (BrainIAC) — all four below baseline ❌
We tried replacing the tiny CNN with **BrainIAC** (`eugenehp/brainiac`), a 3D ViT pretrained
on 32k+ brain MRIs — the only brain-MRI-specific 3D encoder on Hugging Face.

| Experiment | argmax | Sinkhorn | Hungarian |
|---|---|---|---|
| BrainIAC frozen, raw volumes | 0.307 | 0.233 | 0.288 |
| BrainIAC frozen, **skull-stripped** (HD-BET) | 0.117 | 0.103 | 0.116 |
| BrainIAC **fine-tuned: adapter head** | 0.261 | 0.268 | 0.288 |
| BrainIAC **fine-tuned: LoRA r=8** | 0.299 | 0.296 | 0.292 |
| *(reference) SliceCLIP + Sinkhorn* | 0.481 | **0.555** | 0.529 |

What we learned (each a useful negative):
- **Frozen zero-shot is weak**, and reranking *hurt* it — the frozen features aren't
  discriminative for same-subject matching (conflict rate 0.7–0.88).
- **Skull-stripping made it worse (~halved)** — surprising. The frozen features partly key on
  skull/field-of-view geometry; removing it *collapsed* the embeddings. So preprocessing
  mismatch was **not** the bottleneck.
- **The frozen feature is near-collapsed** (off-diagonal cosine ~0.9999); fine-tuning needs
  per-dimension feature standardization just to train at all.
- **Fine-tuning (head and LoRA) overfits dataset1 and does not transfer.** Train MRR climbs to
  0.5–0.9 but LB stays ~0.27–0.30, because the LB is dominated by ds2/ds3 (no labels there).
- **Conclusion:** the whole BrainIAC line is a dead end. Closing the gap needs ds2/ds3 signal,
  not more ds1 adaptation. *(PRs #5 frozen, #6 skull-strip, #7 fine-tune.)*

### 4.5 Cross-attention reranker (D2) — below baseline ❌
A small cross-attention head over SliceCLIP conv tokens, trained on ds1 (PR #8). Scored
**0.464** Sinkhorn vs 0.555. Two diagnosed causes: the conv tokens barely vary across subjects
(std ~0.006; the signal lives in the projection MLP, not the tokens), and the ds1-trained
refinement scrambles ds3. The synthetic-ds2 proxy correctly predicted Δ≈0 before submitting.

### 4.6 ds2-style augmentation (F1) — works on ds2, but a single encoder trades off ⚖️
Train SliceCLIP with independent rigid(3D)+elastic deformation of query/target, reproducing
ds2 (PR #9). **Offline synthetic-ds2 MRR 0.11 → 0.35 (~3×) — the mechanism works.** But one
shared encoder handling clean *and* deformed inputs pays the ds2 gain back on ds1:

| Dataset | baseline | augmented (prob 0.25) |
|---|---|---|
| ds1 | 0.896 | 0.704 |
| ds2 | 0.120 | **0.335** |
| ds3 | 0.649 | 0.607 |
| overall | **0.555** | 0.549 (ties) |

Lesson: don't force one encoder to do both — route per dataset, or (better) use a method that
is robust by construction (→ MIND, §4.6 superseded this).

### 4.7 MIND modality-invariant retrieval — ✅ BIG WIN (0.801, new best)
A **training-free** method (MIND descriptor, Heinrich 2012): per-voxel local self-similarity
over a 6-neighbourhood via 3D box-convolution on GPU. Modality-invariant by construction, so
the T1↔T2 match is found with **no training** — sidestepping the ds1 overfitting that killed
every learned method. Dissimilarity = mean |MIND_q − MIND_t|, then Sinkhorn. *(PR #10.)*

- **Public LB 0.80088** (+0.25 over the prior 0.555 best).
- Offline: aligned proxy MRR **~1.0**; synthetic-ds2 **~0.44** (correctly flagged the ds2 weakness).
- Real-pool conflict rates: ds1 0.04–0.05, ds3 0.00–0.12 (near-perfect bijection), ds2 0.75–0.83.
- **It beats the augmented SliceCLIP even on ds2** (~0.40 vs 0.335), so it supersedes the §4.6
  routing idea entirely.
- **Remaining gap = ds2 only.** MIND needs spatial correspondence, which ds2's independent
  deformation breaks (COM-alignment barely helped). Next: real rigid/affine pre-registration
  of query↔candidate before MIND on ds2 (🔄 in flight).

---

## 5. Improvement axes

Rough effort/impact are guesses to help prioritize, not promises.
**Done:** A1 (Sinkhorn), G1 (MIND — our 0.801 best). **Confirmed dead ends:** the whole BrainIAC
line (C1) and the cross-attention reranker (D2). **Top lever now: G1b — ds2 pre-registration
before MIND**, which is the entire remaining gap to the leader.

### A. Assignment / ranking structure  *(we already started here)*
- **A1. Sinkhorn vs Hungarian reranking — ✅ DONE** (Sinkhorn best, 0.555).
- **A2. Sinkhorn temperature (`tau`) sweep** — we used default tau=10. Tune on val pools. *(low effort / medium)*
- **A3. Per-dataset strategy** — Hungarian on clean ds1, Sinkhorn on deformed ds2/ds3; conflict rates differ per pool. *(low / medium)*
- **A4. Reciprocal / mutual-nearest-neighbour reranking** — boost candidates that also rank the query highly. *(low / low-med)*
- **A5. Confidence-weighted assignment** — blend raw similarity with assignment posterior instead of hard reorder. *(med / med)*

### B. Image representation  *(baseline discards almost all the volume)*
- **B1. Use more slices** than 3 (e.g. 8–16), or all of them. *(low / medium)*
- **B2. Full 3D CNN encoder** — MI300X has 192 GB, so 3D fits easily. *(high / high)*
- **B3. Multi-plane** axial + coronal + sagittal slices. *(low-med / medium)*
- **B4. Higher resolution** than 96×96. *(low / low-med)*

### C. Encoders / pretrained models
- **C1. Pretrained medical foundation models — ❌ TRIED (BrainIAC), dead end** (see §4.4: frozen, skull-strip, adapter, LoRA all below baseline). A *different* pretrained encoder could still work, but BrainIAC's near-collapsed features + ds1-only overfitting make this low-priority now.
- **C2. ImageNet-pretrained 2D backbones** (ResNet/ViT) fine-tuned, instead of the tiny CNN. *(med / med-high)*
- **C3. Self-supervised pretraining** on the unlabeled ds2/ds3 images. *(high / med-high)*
- **C4. Learnable temperature** — `similarity_scale` is fixed at 5.0; make it a parameter. *(low / low)*

### D. Architecture
- **D1. Single shared encoder** (modality is the only difference) or modality-conditioned encoder. *(low-med / med)*
- **D2. Cross-attention** between query and candidate (reranker on top of bi-encoder) — 🔄 **in progress**. *(high / med-high)*

### E. Training objective & negatives
- **E1. Much larger batch** → more in-batch negatives (MI300X can hold huge batches). *(low / med)*
- **E2. Hard-negative mining** beyond in-batch. *(med / med)*
- **E3. LR schedule + early stopping on a local val-MRR** instead of fixed 500 epochs. *(low / low-med)*

### F. Generalization to ds2/ds3  *(the biggest score lever — we have no labels there)*
- **F1. Train-time deformation augmentation** — apply random rigid (rotation/translation)
  + elastic warps **independently** to query and target, mimicking ds2's construction.
  Directly trains for the deformed setting we score worst on. *(med / high)* ← 🔄 **in progress** (the most promising lever; validate on a synthetic-ds2 set first)
- **F2. Intensity standardization / histogram matching / percentile clipping** for scanner robustness. *(low / med)*
- **F3. Skull stripping / brain extraction** to drop non-brain variation. *(med / med)*
- **F4. Test-time augmentation** — embed several augmented views, average. *(low / low-med)*

### G. Classical / registration-based signals  *(no training; strong for ds2/ds3)*
- **G1. Modality-invariant descriptor — ✅ DONE, our BEST (0.801).** MIND (a stronger cousin of
  NMI) ranked by descriptor distance + Sinkhorn. Near-perfect on aligned ds1/ds3. *(see §4.7)*
- **G1b. ds2 pre-registration before MIND — 🔄 IN FLIGHT, the top lever now.** Rigid/affine
  align query↔candidate first so MIND's correspondence assumption holds on ds2. *(med / HIGH — the whole remaining gap)*
- **G2. Exploit alignment** — ds1 is on a common grid and ds3 targets are resampled into
  query space; voxel-overlap / deformable-registration cost as a feature. *(med-high / med)*

### H. Ensembling & robustness
- **H1. Ensemble** multiple seeds / slice sets / models by averaging similarity matrices —
  also directly addresses the variance warning in §4. *(low-med / med)*
- **H2. Blend learned similarity + NMI (G1) + reranking (A).** *(med / med-high)*

### I. Validation methodology
- **I1. Local hold-out from the 350 ds1 pairs** to get an offline MRR signal before
  spending LB submissions (limit 100/team/day; public LB = val rows only). *(low / enabler)*
- **I2. Guard against overfitting the public (val) LB** — private ranking is the test rows. *(low / risk-reduction)*

---

## 6. Reproduction

**Environment** (GPU server, AMD Instinct MI300X, ROCm):
- `~/venv` with `torch==2.10.0+rocm7.0`, `monai`, `nibabel`, `tqdm`, `scipy`, `numpy`.
- Data extracted at `~/medretrieval/data/` (3 datasets, 1,454 volumes).

**Baseline:**
```bash
# from ~/medretrieval, see slice_clip_baseline.py docstring for the full flag list
~/venv/bin/python slice_clip_baseline.py --data-root ~/medretrieval/data ... --out submission.csv
```

**Reranking (produces argmax / sinkhorn / hungarian submissions):**
```bash
DATA_ROOT=~/medretrieval/data PYTHON=~/venv/bin/python ./run_rerank.sh
# writes rerank_out/{argmax,sinkhorn,hungarian}_submission.csv + conflict diagnostics
```

**Submit:**
```bash
kaggle competitions submit -c ehl-paris-medical-image-retrieval -f <file>.csv -m "<msg>"
```
