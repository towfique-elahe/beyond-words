"""
Phase 1 - Baseline: cached Whisper-small encoder embeddings + weighted MLP head.

Two stages so your 6 GB GPU never holds the big model and the classifier at once:

  Stage A (cache):  run Whisper-small's ENCODER over every clip once, mean-pool
                    the hidden states to a single 768-d vector, save to embeddings.npz.
                    VRAM use is small and you only pay this cost once.

  Stage B (train):  load cached vectors, train a small MLP with class weights
                    (handles your Formal/Khulna imbalance), evaluate with
                    macro-F1 + confusion matrix on the TEST split.

Usage:
    python baseline_whisper.py cache  --manifest manifest_split.csv --out embeddings.npz
    python baseline_whisper.py train  --emb embeddings.npz --epochs 60
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

MODEL = "openai/whisper-small"        # encoder hidden size = 768
SR = 16000


def pick_device():
    """cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------- Stage A: cache -----------------------------
def cache(args):
    import librosa
    from transformers import WhisperFeatureExtractor, WhisperModel

    device = pick_device()
    fe = WhisperFeatureExtractor.from_pretrained(MODEL)
    model = WhisperModel.from_pretrained(MODEL).to(device).eval()
    encoder = model.encoder

    df = pd.read_csv(args.manifest)
    labels = sorted(df["district"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}

    X, y, splits = [], [], []
    rows = df.to_dict("records")
    with torch.no_grad():
        for i in tqdm(range(0, len(rows), args.batch), desc="embedding"):
            chunk = rows[i:i + args.batch]
            wavs, kept = [], []
            for r in chunk:
                try:
                    wavs.append(librosa.load(r["filepath"], sr=SR, mono=True)[0])
                    kept.append(r)
                except Exception as e:
                    print("skip", r["filepath"], e)
            if not wavs:
                continue
            # Whisper expects 30s log-mel; FE pads/truncates automatically.
            feats = fe(wavs, sampling_rate=SR, return_tensors="pt").input_features.to(device)
            out = encoder(feats).last_hidden_state      # (B, T, 768)
            vecs = out.mean(dim=1).cpu().numpy()
            for r, v in zip(kept, vecs):
                X.append(v)
                y.append(lab2id[r["district"]])
                splits.append(r["split"])

    np.savez_compressed(args.out,
                        X=np.stack(X), y=np.array(y),
                        splits=np.array(splits), labels=np.array(labels))
    print(f"Saved {len(X)} embeddings -> {args.out}  (dim={X[0].shape[0]})")


# ----------------------------- Stage B: train -----------------------------
class MLP(nn.Module):
    def __init__(self, d_in, n_cls, hidden=256, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def train(args):
    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    d = np.load(args.emb, allow_pickle=True)
    X, y, splits, labels = d["X"], d["y"], d["splits"], list(d["labels"])
    if args.remap_split:
        # Re-assign splits from another manifest (e.g. the strict speaker-
        # clustered split). Embeddings were cached in manifest row order with
        # zero skips, so rows align 1:1.
        remap = pd.read_csv(args.remap_split)
        assert len(remap) == len(y), (
            f"row count mismatch: {args.remap_split} has {len(remap)} rows, "
            f"embeddings have {len(y)}")
        splits = remap["split"].values
    device = args.device if args.device else (pick_device())

    # standardize features on train stats
    tr = splits == "train"
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    Xn = (X - mu) / sd

    def tensor(mask):
        return (torch.tensor(Xn[mask], dtype=torch.float32).to(device),
                torch.tensor(y[mask], dtype=torch.long).to(device))

    Xtr, ytr = tensor(splits == "train")
    Xva, yva = tensor(splits == "val")
    Xte, yte = tensor(splits == "test")

    # inverse-frequency class weights
    counts = np.bincount(y[splits == "train"], minlength=len(labels))
    w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)),
                     dtype=torch.float32).to(device)

    model = MLP(X.shape[1], len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss(weight=w)

    best_va, best_state = -1, None
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = lossf(model(Xtr), ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            va_pred = model(Xva).argmax(1).cpu().numpy()
        va_f1 = f1_score(yva.cpu().numpy(), va_pred, average="macro")
        if va_f1 > best_va:
            best_va, best_state = va_f1, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep{ep:3d}  loss {loss.item():.3f}  val_macroF1 {va_f1:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        te_pred = model(Xte).argmax(1).cpu().numpy()
    yte_np = yte.cpu().numpy()

    print("\n==== TEST ====")
    print("macro-F1:", round(f1_score(yte_np, te_pred, average="macro"), 4))
    print(classification_report(yte_np, te_pred, target_names=labels, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(yte_np, te_pred),
                       index=labels, columns=labels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cache")
    c.add_argument("--manifest", default="manifest_split.csv")
    c.add_argument("--out", default="embeddings.npz")
    c.add_argument("--batch", type=int, default=16)
    c.set_defaults(func=cache)

    t = sub.add_parser("train")
    t.add_argument("--emb", default="embeddings.npz")
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--remap-split", default="",
                   help="manifest CSV whose 'split' column overrides the cached one")
    t.add_argument("--device", default="",
                   help="cpu|mps|cuda (default: auto). cpu is fully deterministic")
    t.set_defaults(func=train)

    args = ap.parse_args()
    args.func(args)
