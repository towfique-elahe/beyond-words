"""
Phase 2 - Fine-tune facebook/wav2vec2-xls-r-300m end-to-end for Bangla DID.

Sized for a free Kaggle/Colab T4 (16 GB VRAM, fp16 + gradient accumulation),
but runs anywhere: device auto-selects cuda > mps > cpu, and mixed precision
is enabled only on CUDA.

Uses the SAME train/val/test split as the Phase 1 baseline via the portable
splits/manifest_split_rel.csv (relative paths), so results are directly
comparable. Class imbalance is handled with inverse-frequency weighted loss;
macro-F1 is the model-selection and headline metric, as in Phase 1.

Usage (Kaggle, dataset attached as input):
    python finetune_xlsr.py \
        --manifest splits/manifest_split_rel.csv \
        --audio-root /kaggle/input/bangla-accent-voice-data \
        --out-dir /kaggle/working

Local smoke test (tiny subset, 1 epoch):
    python finetune_xlsr.py \
        --manifest splits/manifest_split_rel.csv \
        --audio-root ~/Documents/bangla_accent_voice_data \
        --limit 48 --epochs 1 --batch 2
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

MODEL = "facebook/wav2vec2-xls-r-300m"
SR = 16000
CLIP_SAMPLES = 5 * SR          # all clips are ~5 s; pad/trim to exactly 5 s


def pick_device():
    """cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------- data -----------------------------
class ClipDataset(Dataset):
    def __init__(self, df, audio_root, lab2id):
        self.rows = df.to_dict("records")
        self.audio_root = audio_root
        self.lab2id = lab2id

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        path = r["filepath"] if "filepath" in r else os.path.join(self.audio_root, r["relpath"])
        wav, sr = sf.read(path, dtype="float32")
        assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
        if wav.ndim > 1:                       # safety; clips are mono already
            wav = wav.mean(axis=1)
        if len(wav) >= CLIP_SAMPLES:
            wav = wav[:CLIP_SAMPLES]
        else:
            wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        # Wav2Vec2 feature extraction = per-utterance zero-mean unit-variance
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        return torch.from_numpy(wav), self.lab2id[r["district"]]


def collate(batch):
    wavs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return wavs, ys


# ----------------------------- eval -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, trues = [], []
    for wavs, ys in loader:
        wavs = wavs.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(input_values=wavs).logits
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(ys.numpy())
    return np.concatenate(trues), np.concatenate(preds)


# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="splits/manifest_split_rel.csv")
    ap.add_argument("--audio-root", default="", help="prepended to manifest relpath")
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2, help="gradient accumulation steps")
    ap.add_argument("--lr-encoder", type=float, default=2e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--warmup", type=float, default=0.1, help="fraction of steps")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="per-split cap for smoke tests")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (slower, less VRAM)")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    args = ap.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    from transformers import Wav2Vec2ForSequenceClassification

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    use_amp = device == "cuda" and not args.no_amp
    print(f"device={device}  amp={use_amp}")

    df = pd.read_csv(args.manifest)
    audio_root = os.path.expanduser(args.audio_root)
    labels = sorted(df["district"].unique())
    lab2id = {l: i for i, l in enumerate(labels)}

    parts = {}
    for split in ["train", "val", "test"]:
        part = df[df["split"] == split]
        if args.limit:
            part = part.groupby("district").head(max(1, args.limit // len(labels)))
        parts[split] = part.reset_index(drop=True)
    print({k: len(v) for k, v in parts.items()})

    pin = device == "cuda"
    loaders = {
        s: DataLoader(ClipDataset(p, audio_root, lab2id), batch_size=args.batch,
                      shuffle=(s == "train"), num_workers=args.workers,
                      collate_fn=collate, pin_memory=pin, drop_last=(s == "train"))
        for s, p in parts.items()
    }

    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels),
        label2id=lab2id, id2label={i: l for l, i in lab2id.items()},
    ).to(device)
    model.freeze_feature_encoder()             # keep the conv front-end frozen
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    # inverse-frequency class weights from the train split
    counts = np.bincount(parts["train"]["district"].map(lab2id), minlength=len(labels))
    w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)),
                     dtype=torch.float32).to(device)
    lossf = nn.CrossEntropyLoss(weight=w)

    head_params = [p for n, p in model.named_parameters()
                   if n.startswith(("projector", "classifier"))]
    enc_params = [p for n, p in model.named_parameters()
                  if not n.startswith(("projector", "classifier")) and p.requires_grad]
    opt = torch.optim.AdamW(
        [{"params": enc_params, "lr": args.lr_encoder},
         {"params": head_params, "lr": args.lr_head}],
        weight_decay=1e-4)

    steps_per_epoch = max(1, len(loaders["train"]) // args.accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(args.warmup * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warmup_steps if s < warmup_steps
        else max(0.0, (total_steps - s) / (total_steps - warmup_steps)))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_va, best_path = -1.0, out_dir / "xlsr_best.pt"

    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for i, (wavs, ys) in enumerate(tqdm(loaders["train"], desc=f"epoch {ep}")):
            wavs, ys = wavs.to(device), ys.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(input_values=wavs).logits
                loss = lossf(logits, ys) / args.accum
            scaler.scale(loss).backward()
            running += loss.item() * args.accum
            if (i + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()

        yt, yp = evaluate(model, loaders["val"], device, use_amp)
        va_f1 = f1_score(yt, yp, average="macro")
        print(f"epoch {ep}  train_loss {running / max(1, len(loaders['train'])):.3f}  "
              f"val_macroF1 {va_f1:.4f}")
        if va_f1 > best_va:
            best_va = va_f1
            torch.save({"state_dict": model.state_dict(), "labels": labels,
                        "val_macroF1": best_va, "epoch": ep}, best_path)
            print(f"  saved new best -> {best_path}")

    # ----------------------------- test -----------------------------
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    yt, yp = evaluate(model, loaders["test"], device, use_amp)
    te_f1 = f1_score(yt, yp, average="macro")

    print(f"\n==== TEST (best epoch {ckpt['epoch']}, val_macroF1 {ckpt['val_macroF1']:.4f}) ====")
    print("macro-F1:", round(te_f1, 4))
    print(classification_report(yt, yp, target_names=labels, digits=3))
    print("Confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(yt, yp), index=labels, columns=labels))

    pred_df = parts["test"].copy()
    pred_df["true"] = [labels[i] for i in yt]
    pred_df["pred"] = [labels[i] for i in yp]
    pred_df.to_csv(out_dir / "test_predictions_xlsr.csv", index=False)
    per_class = f1_score(yt, yp, average=None)
    json.dump({"model": MODEL, "seed": args.seed, "epochs": args.epochs,
               "best_epoch": int(ckpt["epoch"]), "val_macroF1": float(ckpt["val_macroF1"]),
               "test_macroF1": float(te_f1),
               "per_class_F1": {l: float(f) for l, f in zip(labels, per_class)}},
              open(out_dir / "metrics_xlsr.json", "w"), indent=2)
    print(f"\nWrote {best_path}, test_predictions_xlsr.csv, metrics_xlsr.json -> {out_dir}")


if __name__ == "__main__":
    main()
