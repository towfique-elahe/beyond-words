"""
Result Analysis chapter figures (thesis Ch. 4), from saved predictions and artifacts.

Writes to reports/figures/ (PDF + PNG):
    results_metric_heatmap     models x {accuracy, macro-P, macro-R, macro-F1}, both splits
    results_bars               accuracy and macro-F1 per model, strict vs clip-level
    results_confusions         confusion matrices: Whisper-medium (strict, clip), Tugstugi (Ben-10) strict
    results_training_curves    train loss + val macro-F1 per epoch, Phase 6 models (from reports/phase6/logs)
    results_perclass_heatmap   per-class F1, classes x models, strict split
    results_forest             strict macro-F1 with 95 % speaker-group CIs, all models
    results_leakage_drop       clip-level minus strict per model
    results_tsne               t-SNE of frozen Whisper-small vs fine-tuned Whisper-medium, strict test clips
    results_tsne_classes       fine-tuned t-SNE, one panel per class
  and, redrawn in the same style from saved predictions (Phases 1-3, same file names as before):
    confusion_matrix_baseline  frozen Whisper-small + MLP, clip-level test
    confusion_matrix_xlsr      fine-tuned XLS-R-300M, clip-level test
    per_class_f1_comparison    per-class F1, baseline vs fine-tuned XLS-R, clip-level
    per_class_f1_baseline      per-class F1 of the baseline, sorted
    training_curve_xlsr        XLS-R fine-tuning: loss and val macro-F1 per epoch (reports/phase2)
    ablation_representation_split  three representations x two split protocols
    data_efficiency            test macro-F1 vs fraction of training clips (reports/phase3)

Usage:
    python make_results_figures.py
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

OUT, R6 = Path("reports/figures"), Path("reports/phase6")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE, REF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#9a9890"
CLASSES = ["Barishal", "Chattogram", "Dhaka", "Formal", "Khulna", "Mymensingh", "Noakhali", "Rajshahi", "Sylhet"]
DISPLAY = {"Formal": "Standard"}      # class name as printed in the thesis; data files keep "Formal"


def disp(c):
    return DISPLAY.get(c, c)


LABELS = [disp(c) for c in CLASSES]
# tag -> (label, predictions path per split, bootstrap key)
MODELS = {
    "mfcc":                  ("MFCC + MLP",                 "MFCC"),
    "frozen_whisper_small":  ("Whisper-small, frozen",      "Whisper"),
    "frozen_whisper_medium": ("Whisper-medium, frozen",     "frozen_whisper_medium"),
    "frozen_tugstugi_ben10": ("Tugstugi (Ben-10), frozen",  "frozen_tugstugi_ben10"),
    "xlsr":                  ("XLS-R-300M, fine-tuned",     "XLS-R"),
    "whisper_medium":        ("Whisper-medium, fine-tuned", "whisper_medium"),
    "tugstugi":              ("Tugstugi, fine-tuned",       "tugstugi"),
    "tugstugi_ben10":        ("Tugstugi (Ben-10), fine-tuned", "tugstugi_ben10"),
    "hishab":                ("Hishab FastConformer, fine-tuned", "hishab"),
}
PRED = {("xlsr", "strict"): "reports/phase3/test_predictions_xlsr_strict.csv",
        ("xlsr", "clip"): "reports/phase2/test_predictions_xlsr.csv"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def preds(tag, split):
    p = PRED.get((tag, split), R6 / split / f"test_predictions_{tag}.csv")
    return pd.read_csv(p)


def metrics_table():
    rows = []
    for tag, (label, _) in MODELS.items():
        for split in ["strict", "clip"]:
            d = preds(tag, split)
            rows.append({"tag": tag, "model": label, "split": split,
                         "accuracy": accuracy_score(d.true, d.pred),
                         "macro_precision": precision_score(d.true, d.pred, average="macro", zero_division=0),
                         "macro_recall": recall_score(d.true, d.pred, average="macro"),
                         "macro_f1": f1_score(d.true, d.pred, average="macro")})
    df = pd.DataFrame(rows)
    df.to_csv(R6 / "metrics_all_models.csv", index=False)
    return df


def metric_heatmap(df):
    cols = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    names = ["Accuracy", "Macro-P", "Macro-R", "Macro-F1"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, split, title in zip(axes, ["clip", "strict"], ["(a) clip-level split", "(b) strict split"]):
        m = df[df.split == split].set_index("model").loc[[v[0] for v in MODELS.values()], cols].values * 100
        im = ax.imshow(m, cmap="Blues", vmin=40, vmax=100, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color=SURFACE if m[i, j] > 82 else INK)
        ax.set_xticks(range(4), names, fontsize=8); ax.set_title(title, loc="left"); ax.grid(False)
        ax.tick_params(length=0)
    axes[0].set_yticks(range(len(MODELS)), [v[0] for v in MODELS.values()], fontsize=8)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="score (%)")
    save(fig, "results_metric_heatmap")


def bars(df):
    labels = [v[0] for v in MODELS.values()]
    y = np.arange(len(labels))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, metric, title in zip(axes, ["accuracy", "macro_f1"], ["(a) accuracy", "(b) macro-F1"]):
        for k, (split, color) in enumerate([("clip", ORANGE), ("strict", BLUE)]):
            v = df[df.split == split].set_index("model").loc[labels, metric].values
            ax.barh(y + (0.19 if k == 0 else -0.19), v, height=0.36, color=color,
                    label="clip-level" if split == "clip" else "strict")
            for yi, vi in zip(y + (0.19 if k == 0 else -0.19), v):
                ax.text(vi + 0.008, yi, f"{vi:.3f}", va="center", fontsize=6.5, color=INK2)
        ax.set_xlim(0.4, 1.08); ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
        ax.set_xlabel("test score")
    axes[0].set_yticks(y, labels, fontsize=8)
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)
    save(fig, "results_bars")


def confusions():
    specs = [("whisper_medium", "strict", "Whisper-medium, strict"),
             ("whisper_medium", "clip", "Whisper-medium, clip-level"),
             ("tugstugi_ben10", "strict", "Tugstugi (Ben-10), strict")]
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.1), gridspec_kw={"wspace": 0.12})
    short = LABELS
    for ax, (tag, split, title) in zip(axes, specs):
        d = preds(tag, split)
        cm = confusion_matrix(d.true, d.pred, labels=CLASSES)
        norm = cm / cm.sum(1, keepdims=True)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(9):
            for j in range(9):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=5.5,
                            color=SURFACE if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(9), short, fontsize=6, rotation=90)
        # class names only on the first panel: on the others they ran into the panel beside them
        ax.set_yticks(range(9), short if ax is axes[0] else [], fontsize=6)
        ax.set_title(title, fontsize=8.5, loc="left"); ax.grid(False); ax.tick_params(length=0)
        ax.set_xlabel("predicted", fontsize=8)
    axes[0].set_ylabel("true", fontsize=8)
    save(fig, "results_confusions")


def training_curves():
    logs = sorted((R6 / "logs").glob("*.csv")) if (R6 / "logs").exists() else []
    if not logs:
        print("no training logs; skipping curves"); return
    runs = {p.stem: pd.read_csv(p) for p in logs}
    order = ["whisper_medium", "tugstugi_ben10", "tugstugi", "hishab"]
    fig, axes = plt.subplots(2, 4, figsize=(7.4, 3.8), sharex="col", gridspec_kw={"hspace": 0.15, "wspace": 0.3})
    for j, tag in enumerate(order):
        for split, color in [("strict", BLUE), ("clip", ORANGE)]:
            d = runs.get(f"{tag}_{split}")
            if d is None: continue
            axes[0, j].plot(d.epoch + 1, d.train_loss, color=color, linewidth=1.4,
                            label="strict" if split == "strict" else "clip-level")
            axes[1, j].plot(d.epoch + 1, d.val_macroF1, color=color, linewidth=1.4, marker="o", markersize=2.5)
        axes[0, j].set_title(MODELS[tag][0].replace(", fine-tuned", ""), fontsize=8.5, loc="left")
        axes[1, j].set_xlabel("epoch", fontsize=8); axes[1, j].set_ylim(0.6, 1.0)
    axes[0, 0].set_ylabel("train loss"); axes[1, 0].set_ylabel("val macro-F1")
    axes[0, 0].legend(fontsize=7, loc="upper right")
    save(fig, "results_training_curves")


def perclass_heatmap():
    pc = pd.read_csv(R6 / "summary_per_class.csv")
    pc = pc[pc.split == "strict"].set_index("tag")
    tags = [t for t in MODELS if t in pc.index]
    m = pc.loc[tags, CLASSES].values.T * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.imshow(m, cmap="Blues", vmin=10, vmax=100, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.0f}", ha="center", va="center", fontsize=7.5,
                    color=SURFACE if m[i, j] > 75 else INK)
    ax.set_yticks(range(9), LABELS, fontsize=8)
    ax.set_xticks(range(len(tags)), [MODELS[t][0].replace(", ", ",\n") for t in tags], fontsize=7, rotation=45, ha="right")
    ax.grid(False); ax.tick_params(length=0); ax.set_title("per-class test F1 (%), strict split", loc="left")
    save(fig, "results_perclass_heatmap")


def forest():
    r = json.load(open(R6 / "bootstrap_ci.json"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True, gridspec_kw={"wspace": 0.08})
    labels = [v[0] for v in MODELS.values()]
    y = np.arange(len(labels))[::-1]
    for ax, split, title in zip(axes, ["clip-level", "strict"], ["(a) clip-level split", "(b) strict split"]):
        pts, lo, hi = [], [], []
        for tag, (label, key) in MODELS.items():
            m = r[split]["models"][key]
            pts.append(m["point"]); lo.append(m["ci95_group"][0]); hi.append(m["ci95_group"][1])
        pts, lo, hi = map(np.array, (pts, lo, hi))
        ax.hlines(y, lo, hi, color=BLUE, linewidth=2)
        ax.scatter(pts, y, color=BLUE, s=28, zorder=3, edgecolor=SURFACE, linewidth=1)
        for yi, p in zip(y, pts):
            ax.text(1.005, yi, f"{p:.3f}", va="center", fontsize=7, color=INK2)
        ax.set_xlim(0.3, 1.06); ax.set_xlabel("test macro-F1 (95 % speaker-group CI)")
        ax.set_title(title, loc="left"); ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, labels, fontsize=8)
    save(fig, "results_forest")


def leakage_drop(df):
    labels = [v[0] for v in MODELS.values()]
    c = df[df.split == "clip"].set_index("model").loc[labels, "macro_f1"].values
    s = df[df.split == "strict"].set_index("model").loc[labels, "macro_f1"].values
    drop = (c - s) * 100
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, drop, color=ORANGE, height=0.6)
    for yi, v in zip(y, drop):
        ax.text(v + 0.3, yi, f"{v:.1f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels, fontsize=8); ax.set_xlabel("macro-F1 drop from clip-level to strict split (points)")
    ax.set_xlim(0, 30); ax.grid(axis="y", visible=False)
    save(fig, "results_leakage_drop")


def tsne_pair():
    ft = np.load("embeddings_whisper_medium_ft_strict_test.npz", allow_pickle=True)
    rel = ft["relpath"]; cls = ft["district"]
    fr = np.load("embeddings.npz", allow_pickle=True)
    man = pd.read_csv("manifest_speaker.csv")
    root = os.path.commonpath(man["filepath"].tolist())
    man_rel = man["filepath"].map(lambda f: os.path.relpath(f, root)).values
    idx = pd.Series(np.arange(len(man_rel)), index=man_rel).loc[rel].values
    Xf = fr["X"][idx]
    assert (np.array(fr["labels"])[fr["y"][idx]] == cls).all()

    def emb(X):
        Xn = (X - X.mean(0)) / (X.std(0) + 1e-6)
        return TSNE(n_components=2, random_state=42, perplexity=30, init="pca").fit_transform(Xn)
    Zf, Zt = emb(Xf), emb(ft["X"])
    np.savez(R6 / "tsne_strict_test.npz", frozen=Zf, finetuned=Zt, district=cls, relpath=rel)

    highlight = {"Barishal": BLUE, "Noakhali": ORANGE, "Formal": AQUA}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), gridspec_kw={"wspace": 0.08})
    for ax, Z, title in zip(axes, [Zf, Zt], ["(a) Whisper-small, frozen (0.793)", "(b) Whisper-medium, fine-tuned (0.919)"]):
        other = ~np.isin(cls, list(highlight))
        ax.scatter(Z[other, 0], Z[other, 1], s=5, color=REF, alpha=0.45, linewidth=0)
        for c, color in highlight.items():
            m = cls == c
            ax.scatter(Z[m, 0], Z[m, 1], s=7, color=color, alpha=0.8, linewidth=0, label=disp(c))
        placed = []
        for c in CLASSES:
            m = cls == c
            cx, cy = np.median(Z[m, 0]), np.median(Z[m, 1])
            span = np.ptp(Z[:, 1])
            while any(abs(cx - px) < 0.18 * np.ptp(Z[:, 0]) and abs(cy - py) < 0.05 * span for px, py in placed):
                cy += 0.05 * span          # nudge a label that would overprint another
            placed.append((cx, cy))
            ax.text(cx, cy, disp(c), fontsize=7, ha="center", va="center", color=INK, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.75))
        ax.set_title(title, loc="left", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].legend(loc="lower left", fontsize=7, markerscale=2, handletextpad=0.2)
    save(fig, "results_tsne")

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    for ax, c in zip(axes.ravel(), CLASSES):
        m = cls == c
        ax.scatter(Zt[~m, 0], Zt[~m, 1], s=4, color=REF, alpha=0.35, linewidth=0)
        ax.scatter(Zt[m, 0], Zt[m, 1], s=7, color=BLUE, alpha=0.85, linewidth=0)
        ax.set_title(f"{disp(c)}  (n = {m.sum()})", loc="left", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "results_tsne_classes")


# ---- Phase 1-3 figures, redrawn in the same style (file names unchanged) ----

def legacy_confusions():
    specs = [("confusion_matrix_baseline", preds("frozen_whisper_small", "clip"), "Frozen Whisper-small + MLP, clip-level test"),
             ("confusion_matrix_xlsr", preds("xlsr", "clip"), "Fine-tuned XLS-R-300M, clip-level test")]
    for name, d, title in specs:
        cm = confusion_matrix(d.true, d.pred, labels=CLASSES)
        norm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(5.0, 4.4))
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        for i in range(9):
            for j in range(9):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7.5,
                        color=SURFACE if norm[i, j] > 0.55 else INK)
        ax.set_xticks(range(9), LABELS, fontsize=8, rotation=90); ax.set_yticks(range(9), LABELS, fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, loc="left", fontsize=9)
        ax.grid(False); ax.tick_params(length=0)
        save(fig, name)


def legacy_perclass():
    base, ft = preds("frozen_whisper_small", "clip"), preds("xlsr", "clip")
    fb = f1_score(base.true, base.pred, labels=CLASSES, average=None)
    ff = f1_score(ft.true, ft.pred, labels=CLASSES, average=None)
    mb, mf = f1_score(base.true, base.pred, average="macro"), f1_score(ft.true, ft.pred, average="macro")
    order = np.argsort(ff)[::-1]
    y = np.arange(9)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.barh(y + 0.19, fb[order], height=0.36, color=ORANGE, label=f"frozen Whisper-small + MLP (macro-F1 {mb:.3f})")
    ax.barh(y - 0.19, ff[order], height=0.36, color=BLUE, label=f"fine-tuned XLS-R-300M (macro-F1 {mf:.3f})")
    for yi, b, f in zip(y, fb[order], ff[order]):
        ax.text(b + 0.005, yi + 0.19, f"{b:.3f}", va="center", fontsize=6.5, color=INK2)
        ax.text(f + 0.005, yi - 0.19, f"{f:.3f}", va="center", fontsize=6.5, color=INK2)
    ax.set_yticks(y, [LABELS[i] for i in order]); ax.set_xlim(0.5, 1.06); ax.set_xlabel("test F1 per class, clip-level split")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=8)
    save(fig, "per_class_f1_comparison")

    order = np.argsort(fb)[::-1]
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    ax.barh(y, fb[order], height=0.6, color=BLUE)
    for yi, b in zip(y, fb[order]):
        ax.text(b + 0.005, yi, f"{b:.3f}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [LABELS[i] for i in order]); ax.set_xlim(0.5, 1.06)
    ax.set_xlabel(f"test F1 per class, frozen Whisper-small + MLP, clip-level split (macro-F1 {mb:.3f})")
    ax.grid(axis="y", visible=False)
    save(fig, "per_class_f1_baseline")


def legacy_training_curve():
    d = pd.read_csv("reports/phase2/training_curve_xlsr.csv")
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    ax.plot(d.epoch + 1, d.loss, color=ORANGE, linewidth=1.6, label="train loss")
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss", color=ORANGE); ax.tick_params(axis="y", colors=ORANGE)
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.plot(d.epoch + 1, d.val_macroF1, color=BLUE, linewidth=1.6, marker="o", markersize=3, label="val macro-F1")
    ax2.set_ylabel("val macro-F1", color=BLUE); ax2.tick_params(axis="y", colors=BLUE); ax2.set_ylim(0, 1.02)
    ax.set_title("XLS-R-300M fine-tuning, clip-level split, 20 epochs", loc="left")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)
    save(fig, "training_curve_xlsr")


def legacy_ablation(df):
    tags = [("mfcc", "MFCC statistics\n+ MLP"), ("frozen_whisper_small", "Frozen Whisper-small\n+ MLP"), ("xlsr", "Fine-tuned\nXLS-R-300M")]
    x = np.arange(3); w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 3.0))
    for k, (split, color, label) in enumerate([("clip", ORANGE, "clip-level split (upper bound)"), ("strict", BLUE, "strict pseudo-speaker split")]):
        v = [df[(df.tag == t) & (df.split == split)].macro_f1.iloc[0] for t, _ in tags]
        xs = x + (k - 0.5) * (w + 0.03)
        ax.bar(xs, v, width=w, color=color, label=label)
        for xi, vi in zip(xs, v):
            ax.text(xi, vi + 0.012, f"{vi:.3f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x, [l for _, l in tags]); ax.set_ylim(0, 1.08); ax.set_ylabel("test macro-F1")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=8)
    save(fig, "ablation_representation_split")


def legacy_data_efficiency():
    d = pd.read_csv("reports/phase3/data_efficiency.csv")
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    for rep, color in [("Frozen Whisper-small", BLUE), ("v1 MFCC stats", ORANGE)]:
        for split, ls in [("clip-level", "-"), ("strict", "--")]:
            g = d[(d.representation == rep) & (d.split == split)].groupby("fraction").test_macroF1
            m, lo, hi = g.mean(), g.min(), g.max()
            ax.fill_between(m.index, lo, hi, color=color, alpha=0.15, linewidth=0)
            name = "frozen Whisper-small" if rep.startswith("Frozen") else "MFCC statistics"
            ax.plot(m.index, m.values, color=color, linestyle=ls, linewidth=1.5, marker="o", markersize=3, label=f"{name} ({split})")
    ax.set_xscale("log"); ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0], ["0.10", "0.25", "0.50", "0.75", "1.00"])
    ax.set_xlabel("fraction of training clips"); ax.set_ylabel("test macro-F1"); ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.5)
    save(fig, "data_efficiency")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = metrics_table()
    metric_heatmap(df); bars(df); confusions(); training_curves(); perclass_heatmap(); forest(); leakage_drop(df)
    legacy_confusions(); legacy_perclass(); legacy_training_curve(); legacy_ablation(df); legacy_data_efficiency()
    tsne_pair()


if __name__ == "__main__":
    main()
