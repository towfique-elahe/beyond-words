"""
Dataset chapter figures (thesis Ch. 3).

Reads manifest_speaker.csv (one row per clip: filepath, district, source_id,
speaker_id, duration_s, split) and the audio itself, and writes to
reports/figures/ (PDF + PNG):
    dataset_class_sizes      clips and hours per class
    dataset_durations        clip-duration histogram + per-class box plot
    dataset_spectrograms     one clip per class: waveform + log-mel spectrogram
    dataset_ltas             long-term average spectrum per class vs corpus mean
    dataset_speaker_groups   pseudo-speaker group sizes (strict split)

Usage:
    python make_dataset_figures.py [--manifest manifest_speaker.csv] [--max-clips 0]
"""

import argparse
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch
from tqdm import tqdm

OUT = Path("reports/figures")
SR = 16000
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # reference palette, slots 1-3
INK, INK2, GRID, SURFACE, REF = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff", "#9a9890"
CLASSES = ["Barishal", "Chattogram", "Dhaka", "Formal", "Khulna",
           "Mymensingh", "Noakhali", "Rajshahi", "Sylhet"]
DISPLAY = {"Formal": "Standard"}      # class name as printed in the thesis; data files keep "Formal"


def disp(c):
    return DISPLAY.get(c, c)


LABELS = [disp(c) for c in CLASSES]

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "pdf.fonttype": 42,
})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / name}.pdf/.png")


def class_sizes(df):
    g = df.groupby("district").agg(clips=("filepath", "count"),
                                   hours=("duration_s", lambda s: s.sum() / 3600)).loc[CLASSES]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.35})
    y = np.arange(len(CLASSES))[::-1]
    a.barh(y, g["clips"], color=BLUE, height=0.6)
    for yi, v in zip(y, g["clips"]):
        a.text(v + 15, yi, f"{v:,}", va="center", fontsize=8, color=INK)
    a.set_yticks(y, LABELS); a.set_xlabel("clips"); a.set_xlim(0, 1950); a.grid(axis="y", visible=False)
    a.set_title("(a) clips per class", loc="left")
    b.barh(y, g["hours"], color=ORANGE, height=0.6)
    for yi, v in zip(y, g["hours"]):
        b.text(v + 0.03, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
    b.set_yticks(y, LABELS); b.set_xlabel("hours of audio"); b.set_xlim(0, 3.0); b.grid(axis="y", visible=False)
    b.set_title("(b) hours per class", loc="left")
    save(fig, "dataset_class_sizes")


def durations(df):
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1, 1.4], "wspace": 0.3})
    a.hist(df["duration_s"], bins=np.arange(1.0, 8.3, 0.25), color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    a.set_yscale("log"); a.set_xlabel("clip duration (s)"); a.set_ylabel("clips (log scale)")
    a.set_title("(a) all clips", loc="left"); a.grid(axis="x", visible=False)
    data = [df.loc[df["district"] == c, "duration_s"].values for c in CLASSES]
    bp = b.boxplot(data, vert=False, widths=0.55, patch_artist=True, showfliers=True,
                   flierprops=dict(marker=".", markersize=3, markerfacecolor=INK2, markeredgecolor="none", alpha=0.5),
                   medianprops=dict(color=INK, linewidth=1.2),
                   boxprops=dict(facecolor=BLUE, alpha=0.35, edgecolor=BLUE), whiskerprops=dict(color=INK2),
                   capprops=dict(color=INK2))
    b.set_yticks(range(1, 10), LABELS); b.set_xlabel("clip duration (s)"); b.invert_yaxis()
    b.set_title("(b) per class", loc="left"); b.grid(axis="y", visible=False)
    save(fig, "dataset_durations")


def pick_examples(df):
    """One test-split clip per class, close to 5 s, deterministic."""
    ex = {}
    for c in CLASSES:
        d = df[(df["district"] == c) & (df["split"] == "test") & (df["duration_s"].between(4.8, 5.2))]
        ex[c] = d.sort_values("filepath").iloc[len(d) // 2]["filepath"]
    return ex


def spectrograms(df):
    ex = pick_examples(df)
    fig = plt.figure(figsize=(7.2, 8.2))
    gs = fig.add_gridspec(9, 3, height_ratios=[1, 3.2, 0.9] * 3, hspace=0.1, wspace=0.12)
    for k, c in enumerate(CLASSES):
        r, col = (k // 3) * 3, k % 3
        wav, sr = sf.read(ex[c], dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        wav = wav[:5 * SR]
        t = np.arange(len(wav)) / SR
        aw = fig.add_subplot(gs[r, col])
        aw.plot(t, wav / (np.abs(wav).max() + 1e-6), color=BLUE, linewidth=0.4)
        aw.set_xlim(0, 5); aw.set_ylim(-1.05, 1.05); aw.axis("off")
        aw.set_title(disp(c), fontsize=9, loc="left", pad=3)
        mel = librosa.feature.melspectrogram(y=wav, sr=SR, n_fft=400, hop_length=160, n_mels=80, fmax=8000)
        db = librosa.power_to_db(mel, ref=np.max)
        asp = fig.add_subplot(gs[r + 1, col])
        asp.imshow(db, origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=0,
                   extent=[0, len(wav) / SR, 0, 80])
        asp.grid(False)
        asp.set_yticks([0, 40, 80]); asp.tick_params(labelsize=7)
        if col: asp.set_yticklabels([])
        else: asp.set_ylabel("mel bin", fontsize=8)
        asp.set_xlabel("time (s)", fontsize=8, labelpad=1)
    save(fig, "dataset_spectrograms")


def ltas_plot(freqs, db, cdb, counts):
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 6.4), sharex=True, sharey=True, constrained_layout=True)
    for ax, c in zip(axes.ravel(), CLASSES):
        ax.plot(freqs / 1000, cdb, color=REF, linewidth=1.2, label="corpus mean")
        ax.plot(freqs / 1000, db[c], color=BLUE, linewidth=1.3, label="this class")
        ax.set_title(f"{disp(c)}  (n = {counts[c]:,})", loc="left", fontsize=9, pad=4)
        ax.set_xlim(0, 8)
    axes[0, 0].legend(fontsize=7, loc="upper right", handlelength=1.5)
    for ax in axes[-1]: ax.set_xlabel("frequency (kHz)")
    for ax in axes[:, 0]: ax.set_ylabel("level (dB)")
    save(fig, "dataset_ltas")

    # difference view: class minus corpus mean, one panel, Standard highlighted
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for c in CLASSES:
        if c == "Formal": continue
        ax.plot(freqs / 1000, db[c] - cdb, color=BLUE, alpha=0.35, linewidth=0.9)
    ax.plot(freqs / 1000, db["Formal"] - cdb, color=ORANGE, linewidth=1.6, label=disp("Formal"))
    ax.plot([], [], color=BLUE, alpha=0.5, linewidth=0.9, label="each regional class")
    ax.axhline(0, color=REF, linewidth=0.8)
    ax.set_xlim(0, 7.5); ax.set_ylim(-9, 9)   # the band-edge roll-off above 7.5 kHz is not dialect
    ax.set_xlabel("frequency (kHz)"); ax.set_ylabel("class LTAS minus corpus mean (dB)")
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "dataset_ltas_diff")


def ltas_from_cache(df):
    d = np.load(OUT / "dataset_ltas.npz")
    counts = df["district"].value_counts().to_dict()
    ltas_plot(d["freqs"], {c: d[c] for c in CLASSES}, d["corpus_db"], counts)


def ltas(df, max_clips):
    """Long-term average spectrum per class: each clip's Welch PSD normalised to
    unit total power (so recording level does not dominate), averaged within the
    class, shown in dB against the corpus-wide average."""
    nper = 1024
    freqs = None
    sums = {c: np.zeros(nper // 2 + 1) for c in CLASSES}
    counts = {c: 0 for c in CLASSES}
    rows = df if not max_clips else df.groupby("district", group_keys=False).head(max_clips)
    for path, c in tqdm(zip(rows["filepath"], rows["district"]), total=len(rows), desc="LTAS"):
        wav, sr = sf.read(path, dtype="float32")
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != SR: wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
        f, p = welch(wav, fs=SR, nperseg=nper, noverlap=nper // 2, window="hann")
        p = p / (p.sum() + 1e-12)
        sums[c] += p; counts[c] += 1; freqs = f
    mean = {c: sums[c] / counts[c] for c in CLASSES}
    corpus = sum(sums.values()) / sum(counts.values())
    db = {c: 10 * np.log10(mean[c] + 1e-15) for c in CLASSES}
    cdb = 10 * np.log10(corpus + 1e-15)
    np.savez(OUT / "dataset_ltas.npz", freqs=freqs, corpus_db=cdb, **{c: db[c] for c in CLASSES})
    ltas_plot(freqs, db, cdb, counts)


def speaker_groups(df):
    src = df["source_id"].fillna("").astype(str)
    grp = src.where(src != "", df["speaker_id"].fillna("").astype(str))
    sizes = grp.value_counts()
    per_class = df.assign(group=grp.values).groupby("district")["group"].nunique().loc[CLASSES]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.35})
    bins = np.array([1, 2, 3, 5, 10, 20, 50, 100, 300])
    a.hist(sizes.values, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    a.set_xscale("log"); a.set_xlabel("clips per pseudo-speaker group (log scale)"); a.set_ylabel("groups")
    a.set_title(f"(a) {len(sizes):,} groups; median {int(sizes.median())}, max {sizes.max()}", loc="left")
    a.grid(axis="x", visible=False)
    y = np.arange(len(CLASSES))[::-1]
    b.barh(y, per_class.values, color=ORANGE, height=0.6)
    for yi, v in zip(y, per_class.values):
        b.text(v + 3, yi, f"{v}", va="center", fontsize=8, color=INK)
    b.set_yticks(y, LABELS); b.set_xlabel("pseudo-speaker groups"); b.set_xlim(0, per_class.max() * 1.18)
    b.set_title("(b) groups per class", loc="left"); b.grid(axis="y", visible=False)
    save(fig, "dataset_speaker_groups")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest_speaker.csv")
    ap.add_argument("--max-clips", type=int, default=0, help="LTAS: cap clips per class (0 = all)")
    ap.add_argument("--ltas-cache", action="store_true", help="re-plot both LTAS figures from the saved npz")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest)
    class_sizes(df)
    durations(df)
    spectrograms(df)
    speaker_groups(df)
    ltas_from_cache(df) if args.ltas_cache else ltas(df, args.max_clips)


if __name__ == "__main__":
    main()
