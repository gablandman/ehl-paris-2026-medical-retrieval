# Cross-Validation & Private-Confidence Report

Cross-modal brain-MRI retrieval (`ehl-paris-medical-image-retrieval`), team
**BRAINROT-LABS**. Metric: mean reciprocal rank (MRR), averaged over
dataset1 / dataset2 / dataset3.

> **Why this report exists.** The public leaderboard scores only the **val**
> query rows; the hidden **private** leaderboard scores the **test** rows. The
> **public board has since saturated at 1.000 across many teams** (our
> MIND + ds2-registration + **Hungarian** also scores **public 1.00000**), so the
> public number is no longer discriminative — **the final ranking is decided
> entirely on the private/test split.** The job here is *de-risking*: estimate
> generalization and pick the final submission deliberately.
>
> **Submissions made:** the argmax control (0.97799), Sinkhorn (0.98796), and
> Hungarian (1.00000) — all the same MIND+registration pipeline, differing only
> in the reranker. Everything else in this report is **offline**.

---

## 0. Recipes, rerankers, method

**Two retrieval recipes** (both training-free; MIND = modality-independent
neighbourhood descriptor, scored by mean-|·| of descriptor fields):

| Recipe | ds1 | ds2 | ds3 |
|---|---|---|---|
| **MIND-only** | plain MIND | plain MIND | plain MIND |
| **MIND + ds2-reg** (our best) | plain MIND | affine pre-register target→query (metric = MIND distance), then MIND | plain MIND |

**Rerankers:** `argmax` (sort each query row by raw similarity); `sinkhorn`
(`exp(τ·S)` → doubly-stochastic via Sinkhorn, τ=10, 50 iters, then sort);
`hungarian` (one-to-one linear assignment, assigned target first).

Because MIND is **training-free**, there is no classic train/test overfit. The
*only* private risk is **distribution shift**: are the hidden test pools harder
than the val pools we tuned on? We attack that from two directions:

1. **Direct, label-free** — per-pool top-1 conflict rate and top1−top2 margin,
   computed on the *actual* val and test pools from the 0.988 run's
   post-registration similarity matrices (`reg_sim_cache.npz`). This is the
   strongest evidence because it is measured on the real test images.
2. **Simulated** — `eval/cv_eval.py` rebuilds proxies from the 350 labelled ds1
   pairs (the only data with gold labels) and reports MRR mean±std:
   - **ds1-real:** repeated random holdout over the real pairs (clean baseline).
   - **synth-ds2:** held-out targets get an INDEPENDENT rigid+elastic deform,
     several seeds, at a **harder** strength than the earlier proxy (which
     saturated at MRR 1.0 while real ds2 still showed 0.10–0.14 conflict).
   - **synth-ds3 (APPROXIMATE):** held-out targets get random cuboid erasure
     (a resection / missing-tissue proxy) plus mild deform. Explicitly a rough
     stand-in — we have no real pre-op→intra-op pairs to validate against.

All registration in CV uses the **same knobs as the 0.988 submission**:
`reg-grid 32`, `reg-iters 100`, identity init (`--no-multi-start`). The Task-1
control reused the **identical** cached matrices, so argmax/sinkhorn/hungarian
differ only in the reranker, never in the similarity.

---

## 1. TASK 1 — argmax control on the real leaderboard

Same pipeline as the 0.988 best, reranked by plain **argmax** instead of
Sinkhorn, on the **identical post-registration similarity matrices** (we cached
`S` per pool so the two differ only in the reranker — a clean apples-to-apples
control; registration was run once).

| Reranker | Public MRR | Δ vs best |
|---|---|---|
| **Sinkhorn** (current best) | **0.98796** | — |
| argmax (control) | 0.97799 | **−0.00997** |

**Reading:** Sinkhorn adds ~0.010 MRR, but argmax alone already reaches 0.978.
So the **load-bearing component is MIND + ds2 registration, not the reranker.**
Sinkhorn is a small, safe top-up — it only re-breaks ties where one gallery
volume is the argmax for several queries, and it cannot move a query whose
top-1 is unique. This matters for private risk: even if Sinkhorn's bijection
assumption is slightly off on a harder test pool, the floor is ~argmax ≈ 0.978.

Where does Sinkhorn's gain come from? From the cached matrices, the number of
queries whose **top-1 changes** under Sinkhorn vs argmax, per pool:

| Pool | conflict | sinkhorn moved top-1 | hungarian moved top-1 |
|---|---|---|---|
| ds1-val (40)  | 0.050 | 0  | 1 |
| ds1-test (100)| 0.040 | 2  | 2 |
| ds2-val (40)  | 0.100 | 3  | 2 |
| **ds2-test (100)** | **0.140** | **23** | 9 |
| ds3-val (20)  | 0.000 | 0  | 0 |
| ds3-test (77) | 0.117 | 5  | 5 |

Almost all of Sinkhorn's work is on **ds2-test** (23/100 top-1s reassigned).
That is exactly the pool the public board does **not** see — encouraging, since
the reranker is most active precisely where the hidden risk lives.

---

## 2. TASK 2 — Cross-validation results

### 2.1 MRR mean ± std (over repeats/seeds), per proxy × recipe × reranker

| Proxy | Recipe | argmax | sinkhorn | hungarian |
|---|---|---|---|---|
| ds1-real (aligned, n=20/2) | MIND-only | 0.990 ± 0.014 | 0.999 ± 0.004 | 1.000 ± 0.000 |
| | **MIND+reg** | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| synth-ds2 (harder deform, n=4) | MIND-only | 0.360 ± 0.021 | 0.658 ± 0.061 | 0.809 ± 0.074 |
| | **MIND+reg** | 0.994 ± 0.013 | 0.994 ± 0.013 | 1.000 ± 0.000 |
| synth-ds3 (masking, n=4) | MIND-only | 0.754 ± 0.101 | 0.964 ± 0.032 | 0.982 ± 0.036 |
| | **MIND+reg** | 0.994 ± 0.013 | 0.994 ± 0.013 | 1.000 ± 0.000 |

(Distributions visualized as mean±std error bars in `eval/cv_plot.png`. Raw
per-repeat values were not persisted, so a true boxplot would need a re-run; for
n=2–20 the error bars are the faithful summary.)

### 2.2 argmax vs sinkhorn vs hungarian — robustness

Two findings, consistent across all proxies:

- **Registration crushes the variance.** On the deformed proxies, MIND-**only**
  is both low and *high-variance* (synth-ds2 argmax 0.360 ± 0.021 → hungarian
  0.809 ± 0.074 — and the reranker swings the mean by ~0.45). Add registration
  and every cell jumps to **0.994–1.000 with near-zero std**. So our actual
  recipe is not just high-scoring but *stable* — the hallmark of something that
  generalizes rather than a lucky split.
- **Reranker order is consistent: hungarian ≥ sinkhorn ≥ argmax**, but once
  registration is applied the gaps shrink to noise on these proxies (0.994 vs
  0.994 vs 1.000). The proxies **saturate**, so they cannot finely separate the
  rerankers — but the *real* leaderboard does: argmax 0.978 < sinkhorn 0.988 <
  hungarian **1.000**. The CV ordering and the LB ordering agree, which is why
  hungarian is the defensible top pick.

### 2.3 val-vs-test: the direct private-health signal

Label-free, measured on the **real** val/test pools (0.988 run). Conflict =
fraction of queries whose argmax target is also some other query's argmax;
margin = mean (top1 − top2) similarity (higher ⇒ more separable ⇒ easier).

| Dataset | val conflict | test conflict | val margin | test margin |
|---|---|---|---|---|
| ds1 | 0.050 | 0.040 | 0.0319 | 0.0275 |
| ds2 | 0.100 | **0.140** | 0.0107 | 0.0090 |
| ds3 | **0.000** | **0.117** | 0.0214 | 0.0173 |

Two clear shifts, **both pointing to test being harder than val**:

- **ds3** is the sharpest: ds3-val (only 20 queries) has *zero* conflict and
  looks perfect, but ds3-test (77 queries) has 0.117 conflict and a lower
  margin. The tiny, clean ds3-val pool is an **optimistic sample** — our public
  ds3 contribution is almost certainly inflated relative to private.
- **ds2** test conflict (0.140) > val (0.100) with a smaller margin too: the
  test deformations are a bit harder / the pool is larger (100 vs 40).
- **ds1** is essentially shift-free (conflict and margin match across splits) →
  ds1 private ≈ ds1 public.

---

## 3. TASK 3 — Recommendation & private-confidence verdict

**Final submission to select: Hungarian (public 1.000) as primary; Sinkhorn
(0.988) as the hedge if the competition allows two final picks.**

Rationale:
- Public is saturated, so the choice is purely about **private/test** behaviour.
- Hungarian got a **perfect val score despite real val conflicts** (it correctly
  resolved the 0.05–0.10 ds1/ds2 ambiguities), and CV ranks it ≥ sinkhorn on the
  harder proxies — so its hard one-to-one assignment is *earning* its score, not
  gaming a quirk.
- The one residual risk is that hard assignment can **cascade** if a test pool is
  much messier than val (a wrong forced match displaces a correct one). Sinkhorn
  is the softer fallback for that case — hence the hedge. And the absolute floor
  is argmax ≈ **0.978**, which needs no bijection assumption at all.

**Private-confidence verdict — cautiously high.**
- **ds1:** shift-free (val≈test conflict & margin) → private ≈ public ≈ 1.0.
- **ds2:** test slightly harder than val (conflict 0.14 vs 0.10), but registration
  reduced ds2 conflict from 0.83→0.14 and CV shows MIND+reg ≈ 0.99 even on a
  *harder-than-real* deformation → expect strong, with minor slippage.
- **ds3:** the soft spot. ds3-val (n=20) is an optimistic, zero-conflict sample;
  ds3-test (n=77, conflict 0.117) is harder, so our public ds3 contribution is
  likely **inflated** relative to private. This is where any private drop will
  come from.
- **Net:** the pipeline degrades *gracefully* (registration rescues hard cases,
  low CV variance), so we expect private to land **high — roughly the high-0.9s —
  not a collapse**, with ds3 the main uncertainty. A perfect public 1.000 will
  almost certainly **not** reproduce exactly on private; somewhere in ~0.96–0.99
  is the honest expectation.

**Bottom line:** select Hungarian (+ Sinkhorn hedge), and treat any private
number in the high-0.9s as success. The result rests on a principled,
training-free pipeline with low measured variance — the opposite of an
overfit-to-public submission.

---

### Appendix — reproduce

- **Task-1 control / cache:** `mind/mind_register.py --mode submit … --reg-iters
  100 --no-multi-start` produces the per-pool similarity matrices; argmax,
  sinkhorn and hungarian are then derived from the *same* matrices.
- **Cross-validation:** `scripts/run_cv_eval.sh` (wraps `eval/cv_eval.py`;
  defaults mirror the 0.988 registration knobs). Writes `runs/cv_results.json`.
- **val/test conflict & margin table (§2.3):** computed directly from the
  cached post-registration matrices of the 0.988 submission.
