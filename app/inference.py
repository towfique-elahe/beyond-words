"""
Shared inference helpers for the Beyond Words demo (Gradio app, JSON API and
the microphone-robustness evaluation).

decode_audio(source)  -> 16 kHz mono float32 (any container ffmpeg can read)
prepare(wav)          -> trimmed, level-normalised waveform
windows(wav)          -> list of 5 s windows (2.5 s hop, no zero-padded tail)
AcousticModel.score   -> class probabilities averaged over the windows

The level normalisation matters: the Whisper feature extractor clips quiet
audio at its log-mel floor, and the training clips (broadcast media) have a
median RMS of about 0.076, whereas a laptop microphone at arm's length records
at a tenth of that. Normalising to the training level recovers most of the
accuracy lost to recording level alone (see eval_mic_sim.py).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass

import librosa
import numpy as np
import soundfile as sf

SR = 16000
WIN = 5 * SR
HOP = WIN // 2
TRAIN_RMS = 0.076          # median RMS of 500 strict-split training clips
TRIM_DB = 30               # silence threshold for leading/trailing trim
MIN_SPEECH = 0.5           # seconds of audio a window must carry to be scored
DISPLAY = {"Formal": "Standard"}   # the checkpoint keeps the corpus label


def decode_audio(source) -> np.ndarray:
    """Bytes, path or (sr, array) -> 16 kHz mono float32 in [-1, 1]."""
    if isinstance(source, tuple):                       # Gradio numpy audio
        sr, wav = source
        wav = np.asarray(wav)
        if wav.dtype.kind == "i":
            wav = wav.astype(np.float32) / float(np.iinfo(wav.dtype).max)
        wav = wav.astype(np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if np.abs(wav).max() > 1.0:                     # int16-scaled floats
            wav = wav / 32768.0
    else:
        if isinstance(source, (bytes, bytearray)):
            try:
                wav, sr = sf.read(io.BytesIO(source), dtype="float32", always_2d=False)
            except Exception:                           # webm/opus, m4a, mp3: go through ffmpeg
                wav, sr = _ffmpeg_decode(bytes(source))
        else:
            wav, sr = librosa.load(source, sr=None, mono=False)
        wav = np.asarray(wav, dtype=np.float32)
        if wav.ndim > 1:
            wav = wav.mean(axis=0 if wav.shape[0] <= 8 else 1)
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _ffmpeg_bin() -> str:
    import shutil
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:                                                # dev machines without ffmpeg on PATH
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise ValueError("ffmpeg is not available to decode this audio format") from e


def _ffmpeg_decode(data: bytes):
    """Decode through a temporary file (not a pipe): mp4/m4a containers need seeking."""
    import subprocess
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
        tmp.write(data)
        path = tmp.name
    try:
        p = subprocess.run(
            [_ffmpeg_bin(), "-nostdin", "-loglevel", "error", "-i", path, "-f", "f32le",
             "-ac", "1", "-ar", str(SR), "pipe:1"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    finally:
        os.unlink(path)
    if p.returncode != 0 or not p.stdout:
        raise ValueError("could not decode the audio: " + p.stderr.decode(errors="ignore")[-200:])
    return np.frombuffer(p.stdout, dtype=np.float32).copy(), SR


def prepare(wav: np.ndarray, trim: bool = True, normalise: bool = True) -> np.ndarray:
    """Trim leading/trailing silence and bring the level to the training RMS."""
    wav = np.asarray(wav, dtype=np.float32)
    if trim and len(wav) > SR // 4:
        trimmed, _ = librosa.effects.trim(wav, top_db=TRIM_DB)
        if len(trimmed) >= SR // 2:
            wav = trimmed
    if normalise:
        rms = float(np.sqrt(np.mean(wav ** 2))) if len(wav) else 0.0
        if rms > 1e-5:
            wav = wav * (TRAIN_RMS / rms)
            peak = float(np.abs(wav).max())
            if peak > 0.99:                             # keep headroom, no clipping
                wav = wav * (0.99 / peak)
    return wav.astype(np.float32)


def windows(wav: np.ndarray) -> list[np.ndarray]:
    """5 s windows with 2.5 s hop; a short tail becomes the last 5 s instead of
    being zero-padded; windows that are mostly silence are dropped."""
    n = len(wav)
    if n <= WIN:
        return [np.pad(wav, (0, WIN - n)) if n < WIN else wav]
    starts = list(range(0, n - WIN + 1, HOP))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    out = []
    for s in starts:
        w = wav[s:s + WIN]
        if np.sum(np.abs(w) > 0.01) >= MIN_SPEECH * SR:   # enough non-silent samples
            out.append(w)
    return out or [wav[starts[0]:starts[0] + WIN]]


@dataclass
class Prediction:
    probs: dict            # label -> probability (display names)
    duration: float
    n_windows: int

    @property
    def top(self):
        return max(self.probs, key=self.probs.get)


class AcousticModel:
    """Fine-tuned Whisper encoder classifier (Hugging Face export)."""

    def __init__(self, model_id: str | None = None, device: str | None = None):
        import torch
        from transformers import WhisperFeatureExtractor, WhisperForAudioClassification
        self.torch = torch
        self.model_id = model_id or os.environ.get("MODEL_ID", "towfique-elahe/beyond-words-whisper-medium-did")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fe = WhisperFeatureExtractor.from_pretrained(self.model_id)
        self.model = WhisperForAudioClassification.from_pretrained(self.model_id).to(self.device).eval()
        raw = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.labels = [DISPLAY.get(l, l) for l in raw]

    def score(self, wav: np.ndarray, trim: bool = True, normalise: bool = True) -> Prediction:
        wav = prepare(wav, trim=trim, normalise=normalise)
        wins = windows(wav)
        probs = []
        with self.torch.no_grad():
            for w in wins:
                x = self.fe(w, sampling_rate=SR, padding="max_length", max_length=WIN,
                            return_tensors="pt").input_features
                logits = self.model(input_features=x.to(self.device)).logits
                probs.append(self.torch.softmax(logits, -1)[0].cpu().numpy())
        p = np.mean(probs, axis=0)
        return Prediction({l: float(v) for l, v in zip(self.labels, p)}, len(wav) / SR, len(wins))
