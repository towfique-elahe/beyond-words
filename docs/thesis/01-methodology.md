# Chapter Draft — Methodology

Draft prose + exact settings for the methodology chapter. Every number here is
traceable to a script or artifact in the repository.

## 4.1 Task formulation

Dialect identification (DID) is posed as 9-way single-label classification: given
a ~5-second audio clip, predict one of {Barishal, Chattogram, Dhaka, Formal,
Khulna, Mymensingh, Noakhali, Rajshahi, Sylhet}. "Formal" is a speech register
(standard/prescribed Bangla) rather than a region; it is retained as a ninth class
deliberately, because distinguishing regionally marked speech from the standard
register is itself of practical interest, and because its known source metadata
makes it the calibration anchor for the strict split (§4.3).

## 4.2 Data pipeline

`build_manifest.py` scans the corpus, validates and standardizes every file to
16 kHz mono PCM-16, and extracts recoverable source identifiers from filenames.
Of 9,305 raw clips, 9,303 pass validation (~13.1 h). Only the Formal class carries
recoverable provenance: 559 of its 1,654 clips encode YouTube video IDs (35 unique
sources). The eight regional classes have sequential filenames with no speaker or
source metadata — the root of the leakage problem this thesis quantifies.

## 4.3 Splitting protocols

Two protocols are used throughout; every experiment reports under one or both.

**Canonical clip-level split** (`build_split.py`, 75/10/15 train/val/test targets,
seed 42): stratified by class; rows with a source ID are split source-disjoint via
`StratifiedGroupKFold`. Realized sizes: 6,977 / 997 / 1,329. Because regional clips
split at clip level, the same speaker or recording may appear on both sides;
scores under this protocol are read as an **upper bound**. The exact split is
committed (`splits/manifest_split_rel.csv`) for reproducibility.

**Strict pseudo-speaker split** (`build_speaker_clusters.py` + `build_split.py`):
each clip is embedded with ECAPA-TDNN (SpeechBrain `spkrec-ecapa-voxceleb`,
192-d). Within each class, embeddings are clustered by agglomerative clustering
(cosine distance, average linkage, no preset cluster count). The distance
threshold is not hand-picked: it is **calibrated on the Formal class**, whose true
source IDs are known, by sweeping 0.30–0.90 and keeping the threshold that
maximizes the adjusted Rand index against the true sources (best: 0.50,
ARI = 0.347). Clustering at 0.50 yields 1,683 pseudo-speaker groups; the grouped
split guarantees **no group crosses train/val/test** (verified programmatically:
0 of 1,683). Realized sizes: 6,931 / 1,045 / 1,327
(`splits/manifest_split_strict_rel.csv`).

The modest calibration ARI is disclosed: pseudo-speakers only partially recover
true sources, so the strict split removes much — not provably all — of the
speaker/channel overlap. It is a lower-cost stand-in for true speaker labels,
which the corpus lacks.

## 4.4 Models

All models share one evaluation protocol: inverse-frequency class-weighted
cross-entropy, model selection by validation macro-F1, test macro-F1 as the
headline metric, seed 42.

**(a) v1 re-implementation — MFCC statistics + MLP** (`baseline_mfcc.py`):
40 MFCCs plus Δ and ΔΔ, each mean- and std-pooled over time (240-d), feeding the
same MLP head as (b). This recreates the spirit of the original hand-crafted
pipeline under the current protocol so representations are compared fairly.

**(b) Frozen Whisper-small embeddings + MLP** (`baseline_whisper.py`):
each clip is encoded once by the frozen Whisper-small encoder; hidden states are
mean-pooled to 768-d and cached. The classifier is a 2-hidden-layer MLP
(768→256→256→9, ReLU, dropout 0.3), AdamW (lr 1e-3, weight decay 1e-4), 60
full-batch epochs on CPU (deterministic), best-validation checkpointing.

**(c) End-to-end fine-tuned XLS-R-300M** (`finetune_xlsr.py`):
`facebook/wav2vec2-xls-r-300m` with a sequence-classification head, convolutional
feature encoder frozen, transformer and head fine-tuned. Two learning-rate groups
(encoder 2e-5, head 1e-4), 10 % linear warmup then linear decay, batch 8 ×
gradient accumulation 2 (effective 16), fp16 mixed precision, 20 epochs,
gradient clipping at 1.0. Inputs are 5 s waveforms, per-utterance normalized.
An 8-epoch pilot (macro-F1 0.865, best epoch last) established that the schedule
needed lengthening; at 20 epochs the validation curve plateaus from epoch ~11.

**Augmentation variant** (`finetune_xlsr.py --augment`): train-split-only waveform
augmentation targeting recording-channel signatures — gain ±6 dB (p=.5), additive
Gaussian noise at 10–30 dB SNR (p=.5), 7-band parametric EQ ±6 dB (p=.3),
telephone-style band-pass 300–3400 Hz (p=.15), pitch shift ±2 semitones (p=.25),
time stretch 0.9–1.1 (p=.25).

## 4.5 Data-efficiency protocol

`data_efficiency.py` retrains the identical MLP head on stratified fractions
{10, 25, 50, 75, 100} % of the train split, 3 seeds per point, for representations
(a) and (b) under both split protocols; evaluation is always on the full fixed
test split. Fine-tuning is excluded (each point would cost a full GPU run).

## 4.6 Compute and reproducibility

Development and all non-fine-tuning experiments ran on an Apple M1 laptop (16 GB;
PyTorch MPS). Fine-tuning ran on Kaggle's free T4 tier (16 GB VRAM), ~2.5 h per
run. The pipeline survived a mid-project hardware migration (GTX 1660 Ti → M1)
with the baseline reproducing within 0.3 macro-F1 points — the split files, seeds,
and one-command pipeline stages are the mechanism. Total GPU consumption for every
result in this thesis: under 10 hours of free-tier compute.
