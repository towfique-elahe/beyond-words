"""
Phase 3 addendum - Bootstrap confidence intervals for test macro-F1.

Every headline number is a single seed on a single test set. This script
quantifies how much of each score (and each model-vs-model gap) is sampling
noise, using the saved test predictions:

  * clip bootstrap:  resample test clips with replacement
  * group bootstrap: resample whole pseudo-speaker / source groups — the honest
                     choice, because clips from the same speaker are correlated
                     and a clip bootstrap understates the variance

MLP-head predictions (MFCC, Whisper) are regenerated with the exact seed-42 CPU
recipe of baseline_whisper.py; XLS-R predictions come from the Kaggle runs.

Usage:
    python bootstrap_ci.py --out reports/phase3/bootstrap_ci.json

Phase 5 lexical / fusion predictions are added with --extra:
    python bootstrap_ci.py --out reports/phase5/bootstrap_ci.json \
        --extra "strict:TF-IDF=reports/phase5/test_predictions_<run>__tfidf.csv"
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from baseline_whisper import MLP

AUDIO_ROOT = os.path.expanduser("~/Documents/bangla_accent_voice_data")


def mlp_test_predictions(emb_path, splits, seed=42, epochs=60):
    """Same recipe as `baseline_whisper.py train --device cpu`; returns labels too."""
    d = np.load(emb_path, allow_pickle=True)
    X, y, labels = d["X"], d["y"], list(d["labels"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = splits == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd
    T = lambda m: (torch.tensor(Xn[m], dtype=torch.float32), torch.tensor(y[m], dtype=torch.long))
    Xtr, ytr = T(splits == "train")
    Xva, yva = T(splits == "val")
    Xte, _ = T(splits == "test")
    counts = np.bincount(y[tr], minlength=len(labels))
    w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), dtype=torch.float32)
    model = MLP(X.shape[1], len(labels))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)
    best_va, best_state = -1, None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        lossf(model(Xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            va = f1_score(yva.numpy(), model(Xva).argmax(1).numpy(), average="macro")
        if va > best_va:
            best_va, best_state = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xte).argmax(1).numpy()
    return [labels[i] for i in pred]


def bootstrap(true, preds, groups, B, rng):
    """preds: {name: array}. Returns per-model CIs and all pairwise-gap CIs."""
    names = list(preds)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    out = {m: {"clip": [], "group": []} for m in names}
    gaps = {f"{a} - {b}": {"clip": [], "group": []}
            for i, a in enumerate(names) for b in names[i + 1:]}
    n = len(true)
    for _ in range(B):
        idx_clip = rng.randint(0, n, n)
        idx_grp = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq))])
        for kind, idx in [("clip", idx_clip), ("group", idx_grp)]:
            s = {m: f1_score(true[idx], preds[m][idx], average="macro") for m in names}
            for m in names:
                out[m][kind].append(s[m])
            for k in gaps:
                a, b = k.split(" - ")
                gaps[k][kind].append(s[a] - s[b])

    def ci(v):
        lo, hi = np.percentile(v, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    res = {m: {"point": round(float(f1_score(true, preds[m], average="macro")), 4),
               "ci95_clip": ci(out[m]["clip"]), "ci95_group": ci(out[m]["group"])}
           for m in names}
    gap_res = {k: {"point": round(float(f1_score(true, preds[k.split(' - ')[0]], average="macro")
                                        - f1_score(true, preds[k.split(' - ')[1]], average="macro")), 4),
                   "ci95_clip": ci(v["clip"]), "ci95_group": ci(v["group"]),
                   "p_gap_le_0_group": round(float(np.mean(np.array(v["group"]) <= 0)), 4)}
               for k, v in gaps.items()}
    return res, gap_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default="reports/phase3/bootstrap_ci.json")
    ap.add_argument("--extra", action="append", default=[], metavar="SPLIT:NAME=CSV",
                    help="extra test-prediction CSV (relpath,pred), e.g. "
                         "strict:TF-IDF=reports/phase5/test_predictions_x__tfidf.csv")
    args = ap.parse_args()
    rng = np.random.RandomState(0)

    spk = pd.read_csv("manifest_speaker.csv")          # clip-level split + pseudo-speakers
    strict = pd.read_csv("manifest_split_strict.csv")  # strict split, same row order
    assert (spk["filepath"].values == strict["filepath"].values).all()
    rel = spk["filepath"].str.slice(len(AUDIO_ROOT) + 1)
    src = spk["source_id"].fillna("").astype(str)
    group = src.where(src != "", spk["speaker_id"].fillna("").astype(str)).values

    xlsr = {
        "clip-level": {"XLS-R": "reports/phase2/test_predictions_xlsr.csv"},
        "strict": {"XLS-R": "reports/phase3/test_predictions_xlsr_strict.csv",
                   "XLS-R+aug": "reports/phase3/test_predictions_xlsr_augment.csv"},
    }
    for spec in args.extra:                            # Phase 5 lexical / fusion runs
        split_name, rest = spec.split(":", 1)
        model, path = rest.split("=", 1)
        xlsr[split_name][model] = path
    report = {"B": args.B, "note": "95% percentile intervals; group = pseudo-speaker or YouTube source"}
    for name, splits in [("clip-level", spk["split"].values), ("strict", strict["split"].values)]:
        te = splits == "test"
        true = spk.loc[te, "district"].values
        preds = {"MFCC": np.array(mlp_test_predictions("features_v1.npz", splits)),
                 "Whisper": np.array(mlp_test_predictions("embeddings.npz", splits))}
        for model, path in xlsr[name].items():
            p = pd.read_csv(path).set_index("relpath")["pred"]
            preds[model] = p.loc[rel[te].values].values
        res, gaps = bootstrap(true, preds, group[te], args.B, rng)
        report[name] = {"n_test": int(te.sum()), "n_groups": int(len(np.unique(group[te]))),
                        "models": res, "gaps": gaps}
        print(f"\n== {name}  (n={te.sum()}, groups={len(np.unique(group[te]))}) ==")
        for m, r in res.items():
            print(f"  {m:10s} {r['point']:.4f}  clip CI {r['ci95_clip']}  group CI {r['ci95_group']}")
        for k, r in gaps.items():
            print(f"  gap {k:20s} {r['point']:+.4f}  clip CI {r['ci95_clip']}  "
                  f"group CI {r['ci95_group']}  P(gap<=0|group) {r['p_gap_le_0_group']}")

    json.dump(report, open(args.out, "w"), indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
