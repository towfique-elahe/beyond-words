"""
Phase 4 - Inference demo: predict the Bangla dialect of an audio file.

Uses the strict-split XLS-R checkpoint by default (the honest model). Accepts
any common audio format/sample rate; audio longer than 5 s is scored in 5-second
windows whose probabilities are averaged.

Usage:
    python predict.py path/to/clip.wav [more.wav ...]
    python predict.py --checkpoint checkpoints/xlsr_best.pt clip.wav
"""

import argparse

import numpy as np
import torch

MODEL = "facebook/wav2vec2-xls-r-300m"
SR = 16000
WIN = 5 * SR


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(ckpt_path, device):
    from transformers import Wav2Vec2ForSequenceClassification

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    labels = list(ckpt["labels"])
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(labels),
        label2id={l: i for i, l in enumerate(labels)},
        id2label=dict(enumerate(labels)))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), labels


def load_audio(path):
    import librosa
    wav, _ = librosa.load(path, sr=SR, mono=True)
    if len(wav) < SR:  # < 1 s is too little signal
        raise ValueError(f"{path}: clip shorter than 1 s")
    return wav


def windows(wav):
    """Non-overlapping 5 s windows (last one zero-padded)."""
    n = max(1, int(np.ceil(len(wav) / WIN)))
    for i in range(n):
        w = wav[i * WIN:(i + 1) * WIN]
        if len(w) < WIN:
            w = np.pad(w, (0, WIN - len(w)))
        yield (w - w.mean()) / (w.std() + 1e-7)


@torch.no_grad()
def predict(model, wav, device):
    batch = torch.stack([torch.from_numpy(w.astype(np.float32)) for w in windows(wav)])
    logits = model(input_values=batch.to(device)).logits
    return torch.softmax(logits, dim=-1).mean(0).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="+", help="audio file(s): wav/mp3/flac/...")
    ap.add_argument("--checkpoint", default="checkpoints/xlsr_strict_best.pt")
    ap.add_argument("--top", type=int, default=3)
    args = ap.parse_args()

    device = pick_device()
    model, labels = load_model(args.checkpoint, device)
    print(f"model: {args.checkpoint}  (device={device})\n")

    for path in args.audio:
        probs = predict(model, load_audio(path), device)
        order = np.argsort(probs)[::-1][:args.top]
        top = "   ".join(f"{labels[i]} {probs[i] * 100:.1f}%" for i in order)
        print(f"{path}\n  -> {top}")


if __name__ == "__main__":
    main()
