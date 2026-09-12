"""
Phase 3 - Pseudo-speaker clustering for a strict, group-disjoint split.

The 8 regional classes have no speaker/source metadata, so their Phase 0/1/2
split is clip-level (disclosed as an upper bound). This script builds
pseudo-speaker groups from ECAPA-TDNN speaker embeddings so build_split.py can
produce a group-disjoint "strict" split for ALL classes:

  Stage A (embed):   ECAPA-TDNN (speechbrain, VoxCeleb) embedding per clip,
                     cached once to ecapa_embeddings.npz (192-d, with relpaths).

  Stage B (cluster): agglomerative clustering (cosine, average linkage) per
                     district. The distance threshold is CALIBRATED on the
                     Formal class, whose true YouTube source IDs are known:
                     we sweep thresholds and keep the one that best recovers
                     the true sources (adjusted Rand index). Regional clips
                     then get speaker_id = <district>_<cluster>.

Usage:
    python build_speaker_clusters.py embed   --manifest manifest_split.csv --out ecapa_embeddings.npz
    python build_speaker_clusters.py cluster --manifest manifest_split.csv --emb ecapa_embeddings.npz \
        --out manifest_speaker.csv

Then:  python build_split.py --manifest manifest_speaker.csv --out manifest_split_strict.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

SR = 16000


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------- Stage A: embed -----------------------------
def embed(args):
    import soundfile as sf
    from speechbrain.inference.speaker import EncoderClassifier

    device = pick_device()
    print(f"device={device}")
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device})

    df = pd.read_csv(args.manifest)
    X, paths = [], []
    batch, batch_paths = [], []

    def flush():
        if not batch:
            return
        wavs = torch.stack([torch.from_numpy(w) for w in batch]).to(device)
        with torch.no_grad():
            e = enc.encode_batch(wavs).squeeze(1).cpu().numpy()
        X.append(e)
        paths.extend(batch_paths)
        batch.clear()
        batch_paths.clear()

    for _, r in tqdm(df.iterrows(), total=len(df), desc="ecapa"):
        wav, sr = sf.read(r["filepath"], dtype="float32")
        assert sr == SR, r["filepath"]
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        # pad/trim to 5 s so clips batch cleanly
        n = 5 * SR
        wav = wav[:n] if len(wav) >= n else np.pad(wav, (0, n - len(wav)))
        batch.append(wav)
        batch_paths.append(r["filepath"])
        if len(batch) == args.batch:
            flush()
    flush()

    np.savez_compressed(args.out, X=np.concatenate(X), filepath=np.array(paths))
    print(f"Saved {len(paths)} ECAPA embeddings -> {args.out}")


# ----------------------------- Stage B: cluster -----------------------------
def cluster(args):
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import adjusted_rand_score
    from sklearn.preprocessing import normalize

    df = pd.read_csv(args.manifest)
    d = np.load(args.emb, allow_pickle=True)
    emb = pd.DataFrame({"filepath": d["filepath"]})
    emb_X = normalize(d["X"])                       # unit norm -> cosine geometry
    idx = {p: i for i, p in enumerate(emb["filepath"])}
    assert set(df["filepath"]) <= set(idx), "embeddings missing for some clips"

    def cluster_at(X, t):
        return AgglomerativeClustering(
            n_clusters=None, distance_threshold=t,
            metric="cosine", linkage="average").fit_predict(X)

    # --- calibrate threshold on Formal (true source_id known) ---
    formal = df[df["source_id"].fillna("").astype(str) != ""]
    Xf = emb_X[[idx[p] for p in formal["filepath"]]]
    truth = formal["source_id"].values
    best_t, best_ari = None, -1
    for t in np.arange(0.30, 0.91, 0.05):
        ari = adjusted_rand_score(truth, cluster_at(Xf, t))
        print(f"  threshold {t:.2f}  ARI vs true sources {ari:.3f}")
        if ari > best_ari:
            best_t, best_ari = t, ari
    t = args.threshold if args.threshold > 0 else best_t
    print(f"Calibrated on Formal ({formal['source_id'].nunique()} true sources): "
          f"threshold={t:.2f} (ARI {best_ari:.3f})")

    # --- cluster each regional district at the calibrated threshold ---
    df["speaker_id"] = ""
    regional = df["source_id"].fillna("").astype(str) == ""
    for dist in sorted(df.loc[regional, "district"].unique()):
        m = regional & (df["district"] == dist)
        X = emb_X[[idx[p] for p in df.loc[m, "filepath"]]]
        labels = cluster_at(X, t)
        df.loc[m, "speaker_id"] = [f"{dist[:3].lower()}_{c:04d}" for c in labels]
        sizes = pd.Series(labels).value_counts()
        print(f"{dist:12s} {m.sum():5d} clips -> {sizes.size:4d} pseudo-speakers "
              f"(median size {int(sizes.median())}, max {sizes.max()}, "
              f"singletons {(sizes == 1).sum()})")

    df.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} — now run:\n"
          f"  python build_split.py --manifest {args.out} --out manifest_split_strict.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed")
    e.add_argument("--manifest", default="manifest_split.csv")
    e.add_argument("--out", default="ecapa_embeddings.npz")
    e.add_argument("--batch", type=int, default=16)
    e.set_defaults(func=embed)

    c = sub.add_parser("cluster")
    c.add_argument("--manifest", default="manifest_split.csv")
    c.add_argument("--emb", default="ecapa_embeddings.npz")
    c.add_argument("--out", default="manifest_speaker.csv")
    c.add_argument("--threshold", type=float, default=-1,
                   help="override the Formal-calibrated distance threshold")
    c.set_defaults(func=cluster)

    args = ap.parse_args()
    args.func(args)
