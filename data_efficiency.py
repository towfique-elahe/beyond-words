"""
Phase 3 - Data-efficiency curve.

How much labeled data does each representation need? Trains the SAME MLP head
(architecture, class weights, schedule as baseline_whisper.py) on stratified
fractions of the train split and evaluates on the full, fixed test split —
for both cached representations (v1 MFCC stats, frozen Whisper-small) under
both split protocols (clip-level, strict pseudo-speaker). Each point is the
mean over --seeds runs; the band in the figure spans min-max.

XLS-R fine-tuning is excluded: each point would cost a full GPU run.

Usage:
    python data_efficiency.py \
        --whisper embeddings.npz --mfcc features_v1.npz \
        --strict-manifest manifest_split_strict.csv \
        --out-csv reports/phase3/data_efficiency.csv \
        --out-fig reports/figures/data_efficiency.png
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from baseline_whisper import MLP

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def run_once(X, y, splits, n_cls, frac, seed, epochs=60):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split

    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    tr_idx = np.where(splits == "train")[0]
    if frac < 1.0:
        tr_idx, _ = train_test_split(tr_idx, train_size=frac, random_state=seed,
                                     stratify=y[tr_idx])
    mu, sd = X[tr_idx].mean(0), X[tr_idx].std(0) + 1e-6
    Xn = (X - mu) / sd

    def T(idx):
        return (torch.tensor(Xn[idx], dtype=torch.float32),
                torch.tensor(y[idx], dtype=torch.long))

    Xtr, ytr = T(tr_idx)
    Xva, yva = T(np.where(splits == "val")[0])
    Xte, yte = T(np.where(splits == "test")[0])

    counts = np.bincount(y[tr_idx], minlength=n_cls)
    w = torch.tensor(counts.sum() / (n_cls * np.maximum(counts, 1)), dtype=torch.float32)

    model = MLP(X.shape[1], n_cls)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    best_va, best_state = -1, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            va = f1_score(yva.numpy(), model(Xva).argmax(1).numpy(), average="macro")
        if va > best_va:
            best_va, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return f1_score(yte.numpy(), model(Xte).argmax(1).numpy(), average="macro"), len(tr_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper", default="embeddings.npz")
    ap.add_argument("--mfcc", default="features_v1.npz")
    ap.add_argument("--strict-manifest", default="manifest_split_strict.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out-csv", default="reports/phase3/data_efficiency.csv")
    ap.add_argument("--out-fig", default="reports/figures/data_efficiency.png")
    args = ap.parse_args()

    strict = pd.read_csv(args.strict_manifest)["split"].values
    rows = []
    for rep, path in [("Frozen Whisper-small", args.whisper), ("v1 MFCC stats", args.mfcc)]:
        d = np.load(path, allow_pickle=True)
        X, y, labels = d["X"], d["y"], list(d["labels"])
        assert len(strict) == len(y)
        for split_name, splits in [("clip-level", d["splits"]), ("strict", strict)]:
            for frac in FRACTIONS:
                for seed in range(args.seeds):
                    f1, n = run_once(X, y, np.asarray(splits), len(labels), frac, seed)
                    rows.append({"representation": rep, "split": split_name,
                                 "fraction": frac, "n_train": n, "seed": seed,
                                 "test_macroF1": f1})
                    print(f"{rep:22s} {split_name:10s} frac={frac:.2f} "
                          f"seed={seed} n={n:5d}  F1={f1:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Frozen Whisper-small": "tab:blue", "v1 MFCC stats": "tab:orange"}
    styles = {"clip-level": "-", "strict": "--"}
    for (rep, spl), g in df.groupby(["representation", "split"]):
        agg = g.groupby("fraction")["test_macroF1"].agg(["mean", "min", "max"])
        ax.plot(agg.index, agg["mean"], styles[spl], color=colors[rep],
                marker="o", ms=4, label=f"{rep} ({spl})")
        ax.fill_between(agg.index, agg["min"], agg["max"], color=colors[rep], alpha=0.12)
    ax.set_xlabel("fraction of training clips")
    ax.set_ylabel("test macro-F1")
    ax.set_xticks(FRACTIONS)
    ax.set_title("Data efficiency: representation × split protocol")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out_fig, dpi=200)
    print(f"\nWrote {args.out_csv} and {args.out_fig}")


if __name__ == "__main__":
    main()
