"""
Phase 2 / Phase 6 - Fine-tune a pretrained speech encoder end-to-end for Bangla DID.

Phase 2 fine-tuned facebook/wav2vec2-xls-r-300m (the default, unchanged).
Phase 6 reuses the identical recipe for other encoders via --model / --family:
    wav2vec2  Wav2Vec2ForSequenceClassification, conv front-end frozen (XLS-R)
    whisper   WhisperForAudioClassification = Whisper ENCODER + projector +
              mean-pool + classifier; the decoder of a seq2seq checkpoint is
              dropped. The 30 s positional table is cut to 5 s (250 positions)
              so a clip costs 1/6 of the compute; conv1/conv2 frozen.
              (openai/whisper-medium, bengaliAI/tugstugi_*_whisper-medium)
    nemo      NVIDIA NeMo ASR model (hishab/hishab_bn_fastconformer): its
              preprocessor + Conformer encoder, masked mean-pool, linear head.
              Needs nemo_toolkit[asr]; Kaggle only.

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
    python finetune_xlsr.py --model openai/whisper-medium --tag whisper_medium ...

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
def make_augmenter():
    """Waveform augmentation to suppress recording-channel cues (train only).
    scipy/librosa-backed transforms only — no ffmpeg/codec dependencies."""
    from audiomentations import (AddGaussianSNR, BandPassFilter, Compose, Gain,
                                 PitchShift, SevenBandParametricEQ, TimeStretch)
    return Compose([
        Gain(min_gain_db=-6.0, max_gain_db=6.0, p=0.5),
        AddGaussianSNR(min_snr_db=10.0, max_snr_db=30.0, p=0.5),
        SevenBandParametricEQ(min_gain_db=-6.0, max_gain_db=6.0, p=0.3),
        BandPassFilter(min_center_freq=300.0, max_center_freq=3400.0, p=0.15),
        PitchShift(min_semitones=-2.0, max_semitones=2.0, p=0.25),
        TimeStretch(min_rate=0.9, max_rate=1.1, p=0.25),
    ])


class ClipDataset(Dataset):
    def __init__(self, df, audio_root, lab2id, augment=None, transform=None):
        self.rows = df.to_dict("records")
        self.audio_root = audio_root
        self.lab2id = lab2id
        self.augment = augment
        self.transform = transform     # family-specific: wav -> model input

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        path = r["filepath"] if "filepath" in r else os.path.join(self.audio_root, r["relpath"])
        wav, sr = sf.read(path, dtype="float32")
        assert sr == SR, f"{path}: expected {SR} Hz, got {sr}"
        if wav.ndim > 1:                       # safety; clips are mono already
            wav = wav.mean(axis=1)
        if self.augment is not None:
            wav = self.augment(samples=wav, sample_rate=SR)
        if len(wav) >= CLIP_SAMPLES:
            wav = wav[:CLIP_SAMPLES]
        else:
            wav = np.pad(wav, (0, CLIP_SAMPLES - len(wav)))
        return self.transform(wav), self.lab2id[r["district"]]


def collate(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return xs, ys


# ----------------------------- model families -----------------------------
# Each classifier is an nn.Module with forward(x) -> logits, a `transform`
# (waveform -> input tensor, run in the DataLoader workers), `head_prefixes`
# (parameter names that get the head learning rate) and `saved_state()` /
# `load_saved()` so XLS-R checkpoints keep the exact Phase 2 format
# (the bare HF state_dict, loadable by predict.py).

class Wav2Vec2Classifier(nn.Module):
    head_prefixes = ("model.projector", "model.classifier")

    def __init__(self, model_id, labels, grad_ckpt):
        super().__init__()
        from transformers import Wav2Vec2ForSequenceClassification
        lab2id = {l: i for i, l in enumerate(labels)}
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            model_id, num_labels=len(labels),
            label2id=lab2id, id2label={i: l for l, i in lab2id.items()})
        self.model.freeze_feature_encoder()    # keep the conv front-end frozen
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()

    @staticmethod
    def transform(wav):
        # Wav2Vec2 feature extraction = per-utterance zero-mean unit-variance
        wav = (wav - wav.mean()) / (wav.std() + 1e-7)
        return torch.from_numpy(wav)

    def forward(self, x):
        return self.model(input_values=x).logits

    def saved_state(self):
        return self.model.state_dict()

    def load_saved(self, state):
        self.model.load_state_dict(state)


class WhisperClassifier(Wav2Vec2Classifier):
    """Whisper encoder + projector + mean-pool + classifier, on 5 s inputs."""
    head_prefixes = ("model.projector", "model.classifier")

    def __init__(self, model_id, labels, grad_ckpt):
        nn.Module.__init__(self)
        from transformers import WhisperFeatureExtractor, WhisperForAudioClassification
        lab2id = {l: i for i, l in enumerate(labels)}
        self.fe = WhisperFeatureExtractor.from_pretrained(model_id)
        self.model = WhisperForAudioClassification.from_pretrained(
            model_id, num_labels=len(labels),
            label2id=lab2id, id2label={i: l for l, i in lab2id.items()})
        enc = self.model.encoder
        # Whisper's encoder demands 30 s of mel frames; keep the pretrained
        # positional rows for the first 5 s only (500 frames -> 250 positions).
        n_pos = CLIP_SAMPLES // self.fe.hop_length // 2
        pos = nn.Embedding(n_pos, enc.embed_positions.embedding_dim)
        pos.weight.data.copy_(enc.embed_positions.weight.data[:n_pos])
        pos.requires_grad_(False)
        enc.embed_positions = pos
        enc.config.max_source_positions = n_pos
        for p in list(enc.conv1.parameters()) + list(enc.conv2.parameters()):
            p.requires_grad = False            # frozen conv front-end, as XLS-R
        if grad_ckpt:
            self.model.gradient_checkpointing_enable()

    def transform(self, wav):
        # raw amplitude in; log-mel scaled per clip by the extractor
        feats = self.fe(wav, sampling_rate=SR, padding="max_length",
                        max_length=CLIP_SAMPLES, return_tensors="pt").input_features
        return feats[0]                        # (n_mels, 500)

    def forward(self, x):
        return self.model(input_features=x).logits


class NemoClassifier(nn.Module):
    """NeMo ASR model's preprocessor + Conformer encoder, masked mean-pool, head."""
    head_prefixes = ("head",)

    def __init__(self, model_id, labels, grad_ckpt):
        super().__init__()
        import nemo
        import nemo.collections.asr as nemo_asr
        from huggingface_hub import hf_hub_download, list_repo_files
        nemo_file = next(f for f in list_repo_files(model_id) if f.endswith(".nemo"))
        asr = nemo_asr.models.ASRModel.restore_from(hf_hub_download(model_id, nemo_file),
                                                    map_location="cpu")
        print(f"NeMo {nemo.__version__}: {type(asr).__name__}; encoder "
              f"d_model={asr.cfg.encoder.d_model} n_layers={asr.cfg.encoder.n_layers}")
        self.preprocessor = asr.preprocessor
        self.encoder = asr.encoder
        self.head = nn.Linear(asr.cfg.encoder.d_model, len(labels))
        self.model_id = model_id

    @staticmethod
    def transform(wav):
        return torch.from_numpy(wav)           # NeMo normalises its own features

    def forward(self, x):
        lens = torch.full((x.shape[0],), x.shape[1], dtype=torch.long, device=x.device)
        feats, flen = self.preprocessor(input_signal=x, length=lens)
        enc, elen = self.encoder(audio_signal=feats, length=flen)   # (B, D, T)
        mask = (torch.arange(enc.shape[2], device=enc.device)[None, :] < elen[:, None])
        mask = mask.to(enc.dtype)[:, None, :]
        pooled = (enc * mask).sum(2) / mask.sum(2).clamp(min=1.0)
        return self.head(pooled)

    def saved_state(self):
        return self.state_dict()

    def load_saved(self, state):
        self.load_state_dict(state)


FAMILIES = {"wav2vec2": Wav2Vec2Classifier, "whisper": WhisperClassifier,
            "nemo": NemoClassifier}


def detect_family(model_id):
    try:
        from transformers import AutoConfig
        return AutoConfig.from_pretrained(model_id).model_type
    except Exception:
        from huggingface_hub import list_repo_files
        if any(f.endswith(".nemo") for f in list_repo_files(model_id)):
            return "nemo"
        raise


# ----------------------------- eval -----------------------------
@torch.no_grad()
def evaluate(model, loader, device, use_amp):
    model.eval()
    preds, trues = [], []
    for xs, ys in loader:
        xs = xs.to(device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(xs)
        preds.append(logits.argmax(-1).cpu().numpy())
        trues.append(ys.numpy())
    return np.concatenate(trues), np.concatenate(preds)


# ----------------------------- train -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL, help="HF checkpoint id")
    ap.add_argument("--family", choices=["auto"] + sorted(FAMILIES), default="auto")
    ap.add_argument("--tag", default="xlsr", help="names the outputs: {tag}_best.pt, "
                    "metrics_{tag}.json, test_predictions_{tag}.csv")
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
    ap.add_argument("--augment", action="store_true",
                    help="waveform augmentation on the train split (audiomentations)")
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing (slower, less VRAM)")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    args = ap.parse_args()

    from sklearn.metrics import classification_report, confusion_matrix, f1_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device()
    use_amp = device == "cuda" and not args.no_amp
    family = detect_family(args.model) if args.family == "auto" else args.family
    print(f"device={device}  amp={use_amp}  model={args.model}  family={family}  tag={args.tag}")

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

    model = FAMILIES[family](args.model, labels, args.grad_ckpt).to(device)

    pin = device == "cuda"
    aug = make_augmenter() if args.augment else None
    loaders = {
        s: DataLoader(ClipDataset(p, audio_root, lab2id,
                                  augment=(aug if s == "train" else None),
                                  transform=model.transform),
                      batch_size=args.batch,
                      shuffle=(s == "train"), num_workers=args.workers,
                      collate_fn=collate, pin_memory=pin, drop_last=(s == "train"))
        for s, p in parts.items()
    }

    # inverse-frequency class weights from the train split
    counts = np.bincount(parts["train"]["district"].map(lab2id), minlength=len(labels))
    w = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)),
                     dtype=torch.float32).to(device)
    lossf = nn.CrossEntropyLoss(weight=w)

    head_params = [p for n, p in model.named_parameters()
                   if n.startswith(model.head_prefixes)]
    enc_params = [p for n, p in model.named_parameters()
                  if not n.startswith(model.head_prefixes) and p.requires_grad]
    print(f"trainable params: encoder {sum(p.numel() for p in enc_params) / 1e6:.1f}M, "
          f"head {sum(p.numel() for p in head_params) / 1e6:.2f}M")
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
    best_va, best_path = -1.0, out_dir / f"{args.tag}_best.pt"

    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        running = 0.0
        for i, (xs, ys) in enumerate(tqdm(loaders["train"], desc=f"epoch {ep}")):
            xs, ys = xs.to(device), ys.to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(xs)
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
            torch.save({"state_dict": model.saved_state(), "labels": labels,
                        "val_macroF1": best_va, "epoch": ep,
                        "model": args.model, "family": family}, best_path)
            print(f"  saved new best -> {best_path}")

    # ----------------------------- test -----------------------------
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_saved(ckpt["state_dict"])
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
    pred_df.to_csv(out_dir / f"test_predictions_{args.tag}.csv", index=False)
    per_class = f1_score(yt, yp, average=None)
    json.dump({"model": args.model, "family": family, "seed": args.seed, "epochs": args.epochs,
               "best_epoch": int(ckpt["epoch"]), "val_macroF1": float(ckpt["val_macroF1"]),
               "test_macroF1": float(te_f1),
               "per_class_F1": {l: float(f) for l, f in zip(labels, per_class)}},
              open(out_dir / f"metrics_{args.tag}.json", "w"), indent=2)
    print(f"\nWrote {best_path}, test_predictions_{args.tag}.csv, "
          f"metrics_{args.tag}.json -> {out_dir}")


if __name__ == "__main__":
    main()
