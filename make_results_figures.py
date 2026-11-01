"""
Results chapter figures (thesis Ch. 5-6), from saved predictions and artifacts.

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
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9), gridspec_kw={"wspace": 0.12})
    short = [c[:4] for c in CLASSES]
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
        ax.set_xticks(range(9), short, fontsize=6, rotation=90); ax.set_yticks(range(9), short, fontsize=6)
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
    ax.set_yticks(range(9), CLASSES, fontsize=8)
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
            ax.scatter(Z[m, 0], Z[m, 1], s=7, color=color, alpha=0.8, linewidth=0, label=c)
        placed = []
        for c in CLASSES:
            m = cls == c
            cx, cy = np.median(Z[m, 0]), np.median(Z[m, 1])
            span = np.ptp(Z[:, 1])
            while any(abs(cx - px) < 0.18 * np.ptp(Z[:, 0]) and abs(cy - py) < 0.05 * span for px, py in placed):
                cy += 0.05 * span          # nudge a label that would overprint another
            placed.append((cx, cy))
            ax.text(cx, cy, c, fontsize=7, ha="center", va="center", color=INK, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE, edgecolor="none", alpha=0.75))
        ax.set_title(title, loc="left", fontsize=9); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    axes[0].legend(loc="lower left", fontsize=7, markerscale=2, handletextpad=0.2)
    save(fig, "results_tsne")

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    for ax, c in zip(axes.ravel(), CLASSES):
        m = cls == c
        ax.scatter(Zt[~m, 0], Zt[~m, 1], s=4, color=REF, alpha=0.35, linewidth=0)
        ax.scatter(Zt[m, 0], Zt[m, 1], s=7, color=BLUE, alpha=0.85, linewidth=0)
        ax.set_title(f"{c}  (n = {m.sum()})", loc="left", fontsize=8.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    save(fig, "results_tsne_classes")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = metrics_table()
    metric_heatmap(df); bars(df); confusions(); training_curves(); perclass_heatmap(); forest(); leakage_drop(df)
    tsne_pair()


if __name__ == "__main__":
    main()
