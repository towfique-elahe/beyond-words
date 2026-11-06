"""
Microphone-robustness check for a dialect classifier.

Takes N strict-test clips per class, passes each through a simulated laptop
microphone (48 kHz capture and back, pre-emphasis, 120-6500 Hz band-pass, low
level, Gaussian noise) and scores the clean and the simulated audio with and
without the inference fixes of app/inference.py (silence trim + RMS
normalisation). Reports macro-F1, accuracy and the share of clips predicted as
Standard for each condition.

    python eval_mic_sim.py --model checkpoints/hf_export/beyond-words-whisper-medium-did --tag thesis
    python eval_mic_sim.py --model checkpoints/hf_export/beyond-words-whisper-medium-did-robust --tag robust

Writes reports/demo/mic_sim_<tag>.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfilt
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent / "app"))
from inference import DISPLAY, SR, AcousticModel  # noqa: E402

AUDIO_ROOT = os.path.expanduser("~/Documents/bangla_accent_voice_data")


def mic_sim(w, rng, level=0.08, noise=0.003):
    w = librosa.resample(w, orig_sr=SR, target_sr=48000)
    w = librosa.resample(w, orig_sr=48000, target_sr=SR)
    w = librosa.effects.preemphasis(w, coef=0.6)
    w = sosfilt(butter(4, [120, 6500], btype="band", fs=SR, output="sos"), w).astype(np.float32)
    w = w / (np.abs(w).max() + 1e-6) * level
    return (w + rng.normal(0, noise, len(w))).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF export directory or hub id")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-class", type=int, default=30)
    ap.add_argument("--manifest", default="splits/manifest_split_strict_rel.csv")
    ap.add_argument("--audio-root", default=AUDIO_ROOT)
    ap.add_argument("--out-dir", default="reports/demo")
    args = ap.parse_args()

    man = pd.read_csv(args.manifest)
    test = pd.concat([d.sample(min(args.per_class, len(d)), random_state=0)
                      for _, d in man[man.split == "test"].groupby("district")])
    model = AcousticModel(args.model)
    rng = np.random.default_rng(0)
    conds = {
        "clean, no fixes": (False, False, False),
        "clean, fixes": (False, True, True),
        "mic, no fixes": (True, False, False),
        "mic, normalise only": (True, False, True),
        "mic, trim + normalise": (True, True, True),
    }
    truth = [DISPLAY.get(d, d) for d in test.district]
    labels = sorted(set(truth))
    preds = {c: [] for c in conds}
    for _, r in test.iterrows():
        w, sr = sf.read(os.path.join(args.audio_root, r.relpath), dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        sim = mic_sim(w, rng)
        for c, (use_mic, trim, norm) in conds.items():
            preds[c].append(model.score(sim if use_mic else w, trim=trim, normalise=norm).top)
    out = {}
    for c, p in preds.items():
        out[c] = {"macro_f1": round(float(f1_score(truth, p, average="macro", labels=labels)), 3),
                  "accuracy": round(float(accuracy_score(truth, p)), 3),
                  "standard_share": round(float(np.mean([x == "Standard" for x in p])), 3)}
        print(f"{c:24s} macro-F1 {out[c]['macro_f1']:.3f}  acc {out[c]['accuracy']:.3f}  Standard share {out[c]['standard_share']:.2f}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    path = Path(args.out_dir) / f"mic_sim_{args.tag}.json"
    json.dump({"model": args.model, "n": int(len(test)), "per_class": args.per_class, "conditions": out},
              open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    main()
