"""
Phase 5 - Figures for the lexical route (thesis + README).

Reads the JSON artifacts in reports/phase5/ and writes, to reports/figures/:
    phase5_asr_dialect.{pdf,png}      what survives transcription, per front-end
    phase5_classification.{pdf,png}   word match / TF-IDF / fusion vs XLS-R (strict)
    phase5_fusion_per_class.{pdf,png} XLS-R vs fusion per class (strict, Ben-10 front-end)

Usage:
    python make_phase5_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R, OUT = Path("reports/phase5"), Path("reports/figures")
FRONTS = [  # (tag, label) in the order used throughout the thesis
    ("tugstugi_ben10", "Tugstugi (Ben-10)"),
    ("tugstugi", "Tugstugi"),
    ("hishab_fastconformer", "Hishab FastConformer"),
    ("w2v_xlsr_bn", "wav2vec2 Bengali"),
    ("whisper_large_v3", "Whisper-large-v3"),
]
# reference palette, first three categorical slots (validated all-pairs, light mode)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def load(path):
    return json.load(open(path, encoding="utf-8"))


def fig_asr_dialect():
    ben = load(R / "ben10_valid_eval.json")["fronts"]
    recall, oov_formal, oov_regional = [], [], []
    for tag, _ in FRONTS:
        recall.append(next(r for r in ben[tag]["table"] if r["district"] == "all")["dialect_recall"])
        t = pd.DataFrame(load(R / f"diagnostics_{tag}.json")["fronts"][tag]["table"]).set_index("district")
        oov_formal.append(t.loc["Formal", "oov"])
        oov_regional.append(t.drop(index="Formal")["oov"].mean())

    labels = [l for _, l in FRONTS]
    y = np.arange(len(labels))[::-1]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True,
                               gridspec_kw={"wspace": 0.08})

    a.barh(y, recall, height=0.55, color=BLUE)
    for yi, v in zip(y, recall):
        a.text(v + 0.006, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    a.set_xlim(0, 0.26)
    a.set_yticks(y, labels)
    a.set_xlabel("dialect-word recall on Ben-10 valid")
    a.set_title("(a) dialect words reproduced", loc="left")
    a.grid(axis="y", visible=False)

    for yi, f, r in zip(y, oov_formal, oov_regional):
        b.plot([f, r], [yi, yi], color=GRID, linewidth=2, zorder=1)
    b.scatter(oov_formal, y, s=40, color=ORANGE, zorder=2, label="Formal clips",
              edgecolor=SURFACE, linewidth=1.5)
    b.scatter(oov_regional, y, s=40, color=BLUE, zorder=2, label="regional clips (mean)",
              edgecolor=SURFACE, linewidth=1.5)
    b.set_xlim(0, 0.6)
    b.set_xlabel("share of transcript words outside standard Bangla")
    b.set_title("(b) on this corpus", loc="left")
    b.grid(axis="y", visible=False)
    b.legend(loc="upper right", fontsize=8, handletextpad=0.3)
    save(fig, "phase5_asr_dialect")


def fig_classification():
    rows = []
    for tag, label in FRONTS:
        res = load(R / f"metrics_{tag}__indomain_{tag}_strict.json")["results"]
        rows.append([res[k]["test_macro_f1"] for k in ("match", "tfidf", "fusion")])
        xlsr = res["xlsr"]["test_macro_f1"]
    rows = np.array(rows)

    labels = [l.replace(" (", "\n(").replace(" Fast", "\nFast").replace("2 B", "2\nB")
              .replace("-large", "\n-large") for _, l in FRONTS]
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    names = ["word matching", "TF-IDF", "fusion with XLS-R"]
    for i, (name, color) in enumerate(zip(names, (ORANGE, AQUA, BLUE))):
        xs = x + (i - 1) * (w + 0.02)
        ax.bar(xs, rows[:, i], width=w, color=color, label=name)
        for xi, v in zip(xs, rows[:, i]):
            ax.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=7, color=INK)
    ax.axhline(xlsr, color=INK2, linewidth=1, linestyle=(0, (4, 3)),
               label=f"XLS-R alone ({xlsr:.3f})")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("test macro-F1 (strict split)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, fontsize=8)
    save(fig, "phase5_classification")


def fig_fusion_per_class():
    res = load(R / "metrics_tugstugi_ben10__indomain_tugstugi_ben10_strict.json")["results"]
    df = pd.DataFrame({k: res[k]["test_f1_per_class"] for k in ("xlsr", "fusion")})
    df = df.assign(gain=df.fusion - df.xlsr).sort_values("gain")
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for yi, (_, r) in zip(y, df.iterrows()):
        ax.plot([r.xlsr, r.fusion], [yi, yi], color=GRID, linewidth=2, zorder=1)
        ax.text(max(r.xlsr, r.fusion) + 0.015, yi, f"{r.gain * 100:+.1f}", va="center",
                fontsize=8, color=INK)
    ax.scatter(df.xlsr, y, s=40, color=ORANGE, zorder=2, label="XLS-R alone",
               edgecolor=SURFACE, linewidth=1.5)
    ax.scatter(df.fusion, y, s=40, color=BLUE, zorder=2, label="XLS-R + TF-IDF fusion",
               edgecolor=SURFACE, linewidth=1.5)
    ax.set_yticks(y, df.index)
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("test F1 per class (strict split); labels: change in points")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower left", fontsize=8, handletextpad=0.3)
    save(fig, "phase5_fusion_per_class")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig_asr_dialect()
    fig_classification()
    fig_fusion_per_class()


if __name__ == "__main__":
    main()
