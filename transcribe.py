"""
Phase 5 - Transcribe every clip with an off-the-shelf Bangla ASR model.

The dataset ships no transcripts, so the lexical (word-level) route starts
here: one CSV of ASR output per front-end, later consumed by build_lexicon.py
and lexical_did.py. No model is fine-tuned; checkpoints are used as released.

Three backends cover the front-ends compared in the thesis:
    whisper  openai/whisper-large-v3, bengaliAI/tugstugi_bengaliai-asr_whisper-medium,
             bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium (Ben-10)
    ctc      arijitx/wav2vec2-xls-r-300m-bengali (greedy CTC, no external LM)
    nemo     hishab/hishab_bn_fastconformer (needs nemo_toolkit[asr]; Kaggle only)

Output is resume-safe: rows already in the CSV are skipped, so a pilot run
(--per-class) is reused by the later full run.

Usage (pilot, 50 clips per class, local MPS):
    python transcribe.py \
        --model bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium \
        --gen-config-from openai/whisper-medium --tag tugstugi_ben10 \
        --audio-root ~/Documents/bangla_accent_voice_data --per-class 50

Usage (full, Kaggle T4):
    python transcribe.py --model openai/whisper-large-v3 --tag whisper_large_v3 \
        --audio-root /kaggle/input/bangla-accent-voice-data --out-dir /kaggle/working
"""

import argparse
import csv
import os
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from tqdm import tqdm

SR = 16000


def pick_device():
    """cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------- data -----------------------------
def load_wav(path):
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:                       # safety; clips are mono already
        wav = wav.mean(axis=1)
    if sr != SR:                           # thesis clips are 16 kHz; Ben-10 may not be
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return wav


def select_rows(df, per_class, limit, seed):
    """Optional stratified subsample (pilot); manifest order is preserved."""
    if per_class:
        keep = df.groupby("district", group_keys=False).sample(
            n=per_class, random_state=seed).index
        df = df.loc[sorted(keep)]
    if limit:
        df = df.head(limit)
    return df


# ----------------------------- backends -----------------------------
def whisper_backend(args, device):
    from transformers import (GenerationConfig, WhisperForConditionalGeneration,
                              WhisperProcessor)

    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    if not getattr(model.generation_config, "lang_to_id", None):
        # Fine-tuned checkpoints ship a stripped generation config. Borrow the
        # language/task token tables from the base model they were tuned from.
        if not args.gen_config_from:
            raise SystemExit(f"{args.model} has no language table in its generation "
                             "config; pass --gen-config-from <base whisper model>")
        model.generation_config = GenerationConfig.from_pretrained(args.gen_config_from)
    # language/task are passed explicitly below; drop any baked-in prompt
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = model.to(device, dtype).eval()

    @torch.no_grad()
    def run(wavs, paths):
        feats = processor(wavs, sampling_rate=SR, return_tensors="pt").input_features
        ids = model.generate(feats.to(device, dtype), language="bn", task="transcribe",
                             num_beams=1, max_new_tokens=128)
        return processor.batch_decode(ids, skip_special_tokens=True)

    return run


def ctc_backend(args, device):
    from transformers import (Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor,
                              Wav2Vec2ForCTC)

    # Loaded separately so a repo that bundles an n-gram LM does not pull in
    # pyctcdecode/kenlm: decoding here is plain greedy CTC.
    extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.model)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(args.model).to(device).eval()

    @torch.no_grad()
    def run(wavs, paths):
        enc = extractor(wavs, sampling_rate=SR, return_tensors="pt", padding=True)
        mask = enc.get("attention_mask")
        logits = model(enc.input_values.to(device),
                       attention_mask=None if mask is None else mask.to(device)).logits
        return tokenizer.batch_decode(logits.argmax(-1).cpu())

    return run


def nemo_backend(args, device):
    import nemo.collections.asr as nemo_asr
    from huggingface_hub import hf_hub_download, list_repo_files

    # NeMo's own HF-hub loader mis-extracts this repo's archive (no
    # model_config.yaml found), so fetch the .nemo file and restore it directly.
    nemo_file = next(f for f in list_repo_files(args.model) if f.endswith(".nemo"))
    path = hf_hub_download(args.model, nemo_file)
    model = nemo_asr.models.ASRModel.restore_from(path, map_location=device).eval()

    def run(wavs, paths):
        hyps = model.transcribe(paths, batch_size=len(paths), verbose=False)
        if isinstance(hyps, tuple):                    # (best, all) in older NeMo
            hyps = hyps[0]
        return [getattr(h, "text", h) for h in hyps]   # Hypothesis in newer NeMo

    return run


BACKENDS = {"whisper": whisper_backend, "ctc": ctc_backend, "nemo": nemo_backend}


# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model id")
    ap.add_argument("--backend", choices=sorted(BACKENDS), default="whisper")
    ap.add_argument("--tag", default="", help="output name; default derived from --model")
    ap.add_argument("--gen-config-from", default="",
                    help="base whisper model to borrow the generation config from")
    ap.add_argument("--manifest", default="splits/manifest_split_strict_rel.csv")
    ap.add_argument("--audio-root", default="", help="prepended to manifest relpath")
    ap.add_argument("--out-dir", default="transcripts")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--per-class", type=int, default=0, help="pilot: N clips per district")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N rows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tag = args.tag or args.model.split("/")[-1].replace("-", "_")
    out = Path(args.out_dir) / f"{tag}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    audio_root = os.path.expanduser(args.audio_root)

    df = select_rows(pd.read_csv(args.manifest), args.per_class, args.limit, args.seed)
    done = set(pd.read_csv(out)["relpath"]) if out.exists() else set()
    rows = [r for r in df.to_dict("records") if r["relpath"] not in done]
    print(f"{args.model} -> {out}  ({len(done)} done, {len(rows)} to do)")
    if not rows:
        return

    device = pick_device()
    run = BACKENDS[args.backend](args, device)
    print(f"device={device}  backend={args.backend}")

    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not done:
            w.writerow(["relpath", "district", "text"])
        for i in tqdm(range(0, len(rows), args.batch)):
            chunk = rows[i:i + args.batch]
            paths = [os.path.join(audio_root, r["relpath"]) for r in chunk]
            texts = run([load_wav(p) for p in paths], paths)
            for r, t in zip(chunk, texts):
                w.writerow([r["relpath"], r["district"], " ".join(t.split())])
            f.flush()


if __name__ == "__main__":
    main()
