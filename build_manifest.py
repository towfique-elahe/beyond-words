"""
Phase 0 - Manifest builder for Beyond-Words (Bangla Dialect Identification).

Scans  ROOT/<District>/*.wav  and writes manifest.csv with columns:
    filepath, district, source_id, speaker_id, sample_rate, duration_s, status

- Extracts a YouTube-style source_id from filenames like "... (3FGaiarksCw).mp3.wav"
  (this mostly affects the Formal class; regional folders have none -> source_id="").
- Validates each file; flags anything that is not ~5s mono 16k so you can decide
  whether to re-standardize (see standardize=True).

Usage:
    python build_manifest.py --root "F:/bangla_accent_voice_data" --out manifest.csv
    python build_manifest.py --root "F:/bangla_accent_voice_data" --out manifest.csv --standardize
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
import librosa
import numpy as np
from tqdm import tqdm

# District folder name -> canonical label. Add/adjust spellings here if your
# folders differ. Keys are matched case-insensitively.
DISTRICTS = {
    "barishal": "Barishal",
    "chottogram": "Chattogram",
    "chattogram": "Chattogram",
    "dhaka": "Dhaka",
    "formal": "Formal",
    "khulna": "Khulna",
    "mymensingh": "Mymensingh",
    "noakhali": "Noakhali",
    "rajshahi": "Rajshahi",
    "shylet": "Sylhet",
    "sylhet": "Sylhet",
}

# YouTube IDs are 11 chars of [A-Za-z0-9_-], usually inside parentheses.
YT_ID = re.compile(r"\(([A-Za-z0-9_-]{11})\)")

TARGET_SR = 16000


def extract_source_id(filename: str) -> str:
    m = YT_ID.search(filename)
    return m.group(1) if m else ""


def probe_audio(path: Path):
    """Return (sample_rate, duration_s, channels) cheaply via soundfile header."""
    info = sf.info(str(path))
    return info.samplerate, info.frames / info.samplerate, info.channels


def standardize_file(path: Path):
    """Load, convert to 16k mono, overwrite in place. Returns new duration."""
    y, sr = librosa.load(str(path), sr=TARGET_SR, mono=True)
    sf.write(str(path), y, TARGET_SR, subtype="PCM_16")
    return len(y) / TARGET_SR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Root folder holding district subfolders")
    ap.add_argument("--out", default="manifest.csv")
    ap.add_argument("--standardize", action="store_true",
                    help="Rewrite every clip to 16k mono PCM16 (modifies files in place)")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"Root not found: {root}")

    rows = []
    folders = [p for p in root.iterdir() if p.is_dir()]
    if not folders:
        sys.exit("No subfolders found under root.")

    for folder in folders:
        key = folder.name.strip().lower()
        if key not in DISTRICTS:
            print(f"[skip] unknown folder: {folder.name}")
            continue
        district = DISTRICTS[key]
        wavs = sorted(folder.glob("*.wav"))
        print(f"[{district}] {len(wavs)} wav files")

        for w in tqdm(wavs, desc=district, leave=False):
            status = "ok"
            try:
                sr, dur, ch = probe_audio(w)
            except Exception as e:
                rows.append(dict(filepath=str(w), district=district, source_id="",
                                 speaker_id="", sample_rate=-1, duration_s=-1,
                                 status=f"unreadable:{e}"))
                continue

            needs_fix = (sr != TARGET_SR) or (ch != 1)
            if args.standardize and needs_fix:
                try:
                    dur = standardize_file(w)
                    sr, ch = TARGET_SR, 1
                    status = "standardized"
                except Exception as e:
                    status = f"standardize_failed:{e}"
            elif needs_fix:
                status = f"needs_resample(sr={sr},ch={ch})"

            rows.append(dict(
                filepath=str(w),
                district=district,
                source_id=extract_source_id(w.name),
                speaker_id="",                       # filled later by clustering (optional)
                sample_rate=sr,
                duration_s=round(float(dur), 3),
                status=status,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    # ---- summary ----
    print("\n==== SUMMARY ====")
    print(df.groupby("district").size().rename("clips"))
    print("\nTotal clips:", len(df))
    print("Total hours:", round(df["duration_s"].clip(lower=0).sum() / 3600, 2))
    bad = df[~df["status"].isin(["ok", "standardized"])]
    if len(bad):
        print(f"\n[!] {len(bad)} files need attention (sr/channels/unreadable). "
              f"Re-run with --standardize to auto-fix.")
    src = df[df["source_id"] != ""]
    if len(src):
        print(f"\nClips with a recoverable source_id: {len(src)} "
              f"(mostly Formal). Distinct sources: {src['source_id'].nunique()}")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
