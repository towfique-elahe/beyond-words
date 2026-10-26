"""
Phase 6 - Figure: every acoustic model under both split protocols.

Reads reports/phase6/summary.csv (summarize_phase6.py) and writes
reports/figures/phase6_models_split.{pdf,png}: one row per model, a dot for
the clip-level score and one for the strict score, joined by a bar whose
length is the leakage drop. Rows are grouped by regime and sorted by the
strict score.

Usage:
    python make_phase6_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("reports/figures")
BLUE, ORANGE = "#2a78d6", "#eb6834"        # reference palette, slots 1 and 2
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
REGIME_ORDER = ["hand-crafted", "frozen", "fine-tuned"]   # bottom to top

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "pdf.fonttype": 42,
})


def main():
    df = pd.read_csv("reports/phase6/summary.csv").dropna(subset=["clip_f1", "strict_f1"])
    df["regime"] = pd.Categorical(df["regime"], REGIME_ORDER)
    df = df.sort_values(["regime", "strict_f1"], ascending=[True, True]).reset_index(drop=True)
    labels = [f"{m}  ({p})" if p != "none" else m for m, p in zip(df["model"], df["pretraining"])]
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.2, 0.42 * len(df) + 1.2))
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([r.strict_f1, r.clip_f1], [yi, yi], color=GRID, linewidth=3, zorder=1)
        ax.text(r.clip_f1 + 0.012, yi, f"{r.strict_f1:.3f} / {r.clip_f1:.3f}", va="center",
                fontsize=7.5, color=INK2)
    ax.scatter(df["clip_f1"], y, s=42, color=ORANGE, zorder=2, label="clip-level split",
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(df["strict_f1"], y, s=42, color=BLUE, zorder=2, label="strict split",
               edgecolor=SURFACE, linewidth=1.5)
    # regime separators
    bounds = df.groupby("regime", observed=True).size().cumsum().values[:-1]
    for b in bounds:
        ax.axhline(b - 0.5, color=GRID, linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlim(0.4, 1.13)
    ax.set_xticks(np.arange(0.4, 1.01, 0.1))
    ax.set_xlabel("test macro-F1 (labels: strict / clip-level)")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8, handletextpad=0.3)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"phase6_models_split.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {OUT}/phase6_models_split.pdf/.png ({len(df)} models)")


if __name__ == "__main__":
    main()
