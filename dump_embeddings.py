"""
Phase 6 - Dump a fine-tuned classifier's pooled encoder embeddings (for t-SNE).

For a Whisper-family checkpoint the vector is the mean over time of the encoder
output (the projector's input, d_model = 1024 for Whisper-medium); for XLS-R it
is the mean over time of the last hidden state. Same 5 s window and transform
as training. Output npz: X, relpath, district, split, labels.

Usage:
    python dump_embeddings.py --checkpoint whisper_medium_best.pt \
        --manifest splits/manifest_split_strict_rel.csv --audio-root <root> \
        --splits test --out embeddings_whisper_medium_ft_strict.npz
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from predict import load_audio, load_model, pick_device, windows


@torch.no_grad()
def pooled(model, x):
    inner = model.model if hasattr(model, "model") else model
    if hasattr(inner, "encoder") and hasattr(inner.encoder, "embed_positions"):   # Whisper
        return inner.encoder(input_features=x).last_hidden_state.mean(1)
    return inner.wav2vec2(input_values=x).last_hidden_state.mean(1)              # XLS-R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", default="splits/manifest_split_strict_rel.csv")
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--splits", default="test", help="comma-separated: train,val,test")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = pick_device()
    model, labels = load_model(args.checkpoint, device)
    df = pd.read_csv(args.manifest)
    df = df[df["split"].isin(args.splits.split(","))].reset_index(drop=True)
    root = os.path.expanduser(args.audio_root)
    X = []
    for i in tqdm(range(0, len(df), args.batch)):
        xs = [model.transform(next(windows(load_audio(os.path.join(root, r)))))
              for r in df["relpath"].iloc[i:i + args.batch]]
        X.append(pooled(model, torch.stack(xs).to(device)).float().cpu().numpy())
    np.savez_compressed(args.out, X=np.concatenate(X), relpath=df["relpath"].values,
                        district=df["district"].values, split=df["split"].values,
                        labels=np.array(labels))
    print(f"wrote {len(df)} x {X[0].shape[1]} -> {args.out}")


if __name__ == "__main__":
    main()
