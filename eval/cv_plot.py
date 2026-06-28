# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "matplotlib>=3.9"]
# ///
"""Mean +/- std of the CV MRR per (proxy, recipe, reranker).
(cv_eval.py stored only mean/std/ci95/n, not raw per-repeat values, so this is
an error-bar chart rather than a true boxplot.)"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (mean, std, n) from cv_results.json
DATA = {
    "ds1-real (aligned)": {
        "MIND-only": {"argmax": (0.9901, 0.0135, 20), "sinkhorn": (0.9991, 0.0042, 20), "hungarian": (1.0000, 0.0, 20)},
        "MIND+reg":  {"argmax": (1.0000, 0.0, 2),      "sinkhorn": (1.0000, 0.0, 2),     "hungarian": (1.0000, 0.0, 2)},
    },
    "synth-ds2 (harder deform)": {
        "MIND-only": {"argmax": (0.3604, 0.0207, 4), "sinkhorn": (0.6577, 0.0611, 4), "hungarian": (0.8091, 0.0740, 4)},
        "MIND+reg":  {"argmax": (0.9938, 0.0125, 4), "sinkhorn": (0.9938, 0.0125, 4), "hungarian": (1.0000, 0.0, 4)},
    },
    "synth-ds3 (masking ~ resection)": {
        "MIND-only": {"argmax": (0.7542, 0.1007, 4), "sinkhorn": (0.9640, 0.0318, 4), "hungarian": (0.9820, 0.0360, 4)},
        "MIND+reg":  {"argmax": (0.9938, 0.0125, 4), "sinkhorn": (0.9938, 0.0125, 4), "hungarian": (1.0000, 0.0, 4)},
    },
}
RERANKERS = ["argmax", "sinkhorn", "hungarian"]
COLORS = {"argmax": "#9aa0a6", "sinkhorn": "#4285f4", "hungarian": "#34a853"}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)
for ax, (proxy, recipes) in zip(axes, DATA.items()):
    xt, xl = [], []
    x = 0
    for recipe in ["MIND-only", "MIND+reg"]:
        for rr in RERANKERS:
            m, s, n = recipes[recipe][rr]
            ax.bar(x, m, yerr=s, capsize=4, color=COLORS[rr],
                   edgecolor="black", linewidth=0.6,
                   error_kw={"elinewidth": 1.3, "ecolor": "black"})
            ax.text(x, min(m + s + 0.02, 1.04), f"{m:.3f}\n±{s:.3f}",
                    ha="center", va="bottom", fontsize=7)
            xt.append(x); xl.append(f"{rr}\n(n={n})")
            x += 1
        x += 0.6  # gap between recipes
    ax.set_title(proxy, fontsize=10)
    ax.set_xticks(xt); ax.set_xticklabels(xl, fontsize=7)
    ax.set_ylim(0, 1.12); ax.axhline(1.0, color="gray", ls=":", lw=0.8)
    ax.grid(axis="y", alpha=0.3)
    # recipe group labels
    ax.text(1, -0.16, "MIND-only", ha="center", transform=ax.get_xaxis_transform(), fontsize=9, weight="bold")
    ax.text(4.6, -0.16, "MIND+reg", ha="center", transform=ax.get_xaxis_transform(), fontsize=9, weight="bold")
axes[0].set_ylabel("MRR (mean ± std)")
fig.suptitle("Cross-validation: MIND retrieval recipes — registration rescues the deformed pools (ds2/ds3)", fontsize=12)
fig.tight_layout(rect=[0, 0.04, 1, 0.97])
out = "/Users/gklajer/Developer/TEST/HACK/ehl-paris-2026-medical-retrieval/samples/cv_plot.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print("saved", out)
