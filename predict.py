"""
Phase 4 - Inference demo: predict the Bangla dialect of an audio file.

Uses the strict-split XLS-R checkpoint by default (the honest model). Accepts
any common audio format/sample rate; audio longer than 5 s is scored in 5-second
windows whose probabilities are averaged.

Usage:
    python predict.py path/to/clip.wav [more.wav ...]
    python predict.py --checkpoint checkpoints/xlsr_best.pt clip.wav

Phase 5 adds --explain: the clip is also transcribed (Ben-10 Tugstugi ASR) and
the words behind the decision are shown, i.e. which words are outside standard
Bangla and how strongly each matches every region's lexicon.
    python predict.py --explain clip.wav
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


@torch.no_grad()
def dump_probs(model, labels, args, device):
    """Class probabilities for every val/test clip of a split manifest, for the
    Phase 5 late fusion. Scores the first 5 s only, as finetune_xlsr.ClipDataset
    does, so the argmax reproduces the saved test predictions."""
    import os

    import pandas as pd
    from tqdm import tqdm

    df = pd.read_csv(args.manifest)
    df = df[df["split"].isin(["val", "test"])].reset_index(drop=True)
    root = os.path.expanduser(args.audio_root)
    probs = []
    for i in tqdm(range(0, len(df), args.batch)):
        wavs = [next(windows(load_audio(os.path.join(root, r))))
                for r in df["relpath"].iloc[i:i + args.batch]]
        batch = torch.from_numpy(np.stack(wavs).astype(np.float32)).to(device)
        probs.append(torch.softmax(model(input_values=batch).logits, dim=-1).cpu().numpy())
    out = pd.concat([df[["relpath", "split"]],
                     pd.DataFrame(np.concatenate(probs), columns=labels)], axis=1)
    out.to_csv(args.dump_probs, index=False)
    print(f"wrote {len(out)} rows -> {args.dump_probs}")


def load_explainer(args, device):
    """ASR front-end + word lists for --explain (first 30 s of a clip)."""
    import json
    from types import SimpleNamespace

    from text_utils import load_vocab
    from transcribe import whisper_backend

    asr = whisper_backend(SimpleNamespace(model=args.asr_model,
                                          gen_config_from=args.gen_config_from), device)
    formal = load_vocab(args.formal, 3)
    lexicon = json.load(open(args.lexicon, encoding="utf-8"))["lexicon"]
    return asr, formal, lexicon


def explain(wav, asr, formal, lexicon, top):
    from lexical_did import match_scores
    from text_utils import tokenize

    text = asr([wav[:30 * SR]], None)[0]
    tokens = tokenize(text)
    nonformal, scores = match_scores(tokens, formal, lexicon)
    print(f"  transcript: {text}")
    print(f"  words: {len(tokens)}   outside standard Bangla: {nonformal * 100:.0f}%")
    for w in dict.fromkeys(t for t in tokens if t not in formal):
        hits = sorted(((words[w], r) for r, words in lexicon.items() if w in words), reverse=True)
        where = "   ".join(f"{r} {p:.2f}" for p, r in hits[:top]) or "in no regional lexicon"
        print(f"    {w:<14} {where}")
    order = [r for r in sorted(scores, key=scores.get, reverse=True)[:top] if scores[r] > 0]
    print("  word match: " + ("   ".join(f"{r} {scores[r] * 100:.1f}%" for r in order)
                              or "no word in any regional lexicon"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="*", help="audio file(s): wav/mp3/flac/...")
    ap.add_argument("--checkpoint", default="checkpoints/xlsr_strict_best.pt")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--dump-probs", default="", help="write val/test probabilities to this CSV")
    ap.add_argument("--manifest", default="splits/manifest_split_strict_rel.csv")
    ap.add_argument("--audio-root", default="", help="prepended to manifest relpath")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--explain", action="store_true", help="also show word-level evidence")
    ap.add_argument("--asr-model",
                    default="bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium")
    ap.add_argument("--gen-config-from", default="openai/whisper-medium")
    ap.add_argument("--formal", default="lexicons/formal_vocab.txt")
    ap.add_argument("--lexicon", default="lexicons/indomain_tugstugi_ben10_strict.json")
    args = ap.parse_args()

    device = pick_device()
    model, labels = load_model(args.checkpoint, device)
    print(f"model: {args.checkpoint}  (device={device})\n")
    if args.dump_probs:
        return dump_probs(model, labels, args, device)

    explainer = load_explainer(args, device) if args.explain else None

    for path in args.audio:
        wav = load_audio(path)
        probs = predict(model, wav, device)
        order = np.argsort(probs)[::-1][:args.top]
        top = "   ".join(f"{labels[i]} {probs[i] * 100:.1f}%" for i in order)
        print(f"{path}\n  -> {top}")
        if explainer:
            explain(wav, *explainer, args.top)


if __name__ == "__main__":
    main()
