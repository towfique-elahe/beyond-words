"""
Phase 3 ablation - v1-style hand-crafted features (MFCC + deltas, mean/std pooled).

Recreates the spirit of the original v1 pipeline (hand-crafted features + small
MLP) so the ablation table compares three representations under ONE protocol:

    v1 MFCC stats  vs.  frozen Whisper-small embeddings  vs.  fine-tuned XLS-R

The cache is written in the same npz layout as embeddings.npz, so training and
evaluation reuse baseline_whisper.py unchanged (same MLP, weights, seed, splits):

    python baseline_mfcc.py cache --manifest manifest_split.csv --out features_v1.npz
    python baseline_whisper.py train --emb features_v1.npz --device cpu
    python baseline_whisper.py train --emb features_v1.npz --device cpu \
        --remap-split manifest_split_strict.csv
"""

import argparse

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

SR = 16000
N_MFCC = 40


def clip_features(wav):
    """40 MFCCs + Δ + ΔΔ, each mean- and std-pooled over time -> 240-d."""
    m = librosa.feature.mfcc(y=wav, sr=SR, n_mfcc=N_MFCC)
    feats = [m, librosa.feature.delta(m), librosa.feature.delta(m, order=2)]
    return np.concatenate([np.concatenate([f.mean(1), f.std(1)]) for f in feats])


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--manifest", default="manifest_split.csv")
    c.add_argument("--out", default="features_v1.npz")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    labels = sorted(df["district"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}

    X, y, splits = [], [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="mfcc"):
        wav, _ = librosa.load(r["filepath"], sr=SR, mono=True)
        X.append(clip_features(wav))
        y.append(lab2id[r["district"]])
        splits.append(r["split"])

    np.savez_compressed(args.out, X=np.stack(X), y=np.array(y),
                        splits=np.array(splits), labels=np.array(labels))
    print(f"Saved {len(X)} feature vectors -> {args.out}  (dim={X[0].shape[0]})")


if __name__ == "__main__":
    main()
