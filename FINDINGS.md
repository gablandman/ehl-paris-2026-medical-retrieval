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
| Rerank — argmax (parity) | 0.48068 | same method as baseline, fresh model (see variance note) |
| Rerank — Hungarian | 0.52866 | hard one-to-one assignment |
| **Rerank — Sinkhorn** | **0.55487** | **best — currently 2nd place** |

Leaderboard at last check: leader **0.88670**, us **0.55487** (2nd), then 0.498, 0.487, 0.414.

**Headroom: ~0.33 MRR to the leader.** Plenty of axes below.

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

### 4.5 In flight
- **F1 — ds2-style augmentation** on SliceCLIP (branch `ds2-geometric-augmentation`).
- **D2 — cross-attention reranker** (branch in progress). Both target the ds2/ds3 gap above.

---

## 5. Improvement axes

Rough effort/impact are guesses to help prioritize, not promises.
**A1 is done.** Suggested next: **F1 (deformation augmentation)** and **A2 (tau sweep)** —
biggest expected impact for least effort, and they target ds2/ds3 where we're weakest.

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
- **G1. Normalized Mutual Information (NMI)** between query and candidate — the classic
  cross-modal registration metric; rank by it directly or blend with learned similarity. *(med / med-high, esp. ds2/ds3)*
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
