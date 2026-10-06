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
All 9,303 `.wav` clips pass validation (~13.1 h); two duplicate `.ogg` files in the
Formal folder are not used. Clips are nominally 5 s (median 5.0 s), but 1,067 fall
outside 4.5–5.5 s (range 1.1–8.0 s). Only the Formal class carries recoverable
provenance: 557 of its 1,652 clips encode YouTube video IDs (35 unique sources). The eight regional classes have sequential filenames with no speaker or
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
result in this thesis: under 10 hours of free-tier compute for the acoustic route
and about 3.5 further T4 hours for the lexical route's transcription runs.

## 4.7 The lexical route

Models (a)–(c) never observe *what* is said: Whisper consumes log-mel frames,
XLS-R the raw waveform. The lexical route asks how much dialect identity is
recoverable from the words alone, and whether it adds to the acoustics. It
reuses the splits, the macro-F1 protocol and the group bootstrap of §4.3–4.4
unchanged, so every number is directly comparable to (a)–(c).

**(i) Transcription** (`transcribe.py`). The corpus has no transcripts, so text
is produced by off-the-shelf Bangla ASR, used as released (no fine-tuning).
The front-ends follow the Ben-10 ASR benchmark (Dipto et al., IJCNLP-AACL
2025) as far as public checkpoints allow: Whisper-large-v3 (zero-shot);
`bengaliAI/tugstugi_bengaliai-asr_whisper-medium`, the Bengali.AI 2023
competition winner, trained on standard colloquial Bangla; the same model
fine-tuned by Bengali.AI on Ben-10 dialect speech
(`…-regional-asr_whisper-medium`); Hishab's FastConformer
(`hishab/hishab_bn_fastconformer`, NeMo, ~18k h of mostly news speech); and
`arijitx/wav2vec2-xls-r-300m-bengali` as a documented stand-in for the paper's
"Wav2Vec2 (SCB)", whose checkpoint is not public. Google's paid API and the
paper's Ben-10-tuned wav2vec2 (no checkpoint) are omitted. Decoding is greedy
throughout (Whisper: language and task forced to Bangla transcription, no prompt,
≤ 128 new tokens; CTC: argmax, no language model). Fine-tuned Whisper
checkpoints ship without a language-token table, so the base model's
generation config is substituted; this is the standard remedy and changes no
weights. Transcription of all 9,303 clips ran on a Kaggle T4 (43 / 45 / 82 / 15
/ 7 min for the five front-ends); a 450-clip pilot on the M1 agreed with the
T4 output on 93.6 % of clips.

**(ii) Normalisation and the standard-Bangla vocabulary** (`text_utils.py`,
`build_lexicon.py formal`). All text — transcripts, reference corpora, word
lists — passes one tokeniser: Unicode NFC, Bangla-letter tokens only (digits,
punctuation and Latin text dropped), word-level normalisation with
`bnunicodenormalizer`, zero-width joiners stripped. The *formal vocabulary* is
the word inventory of OOD-Speech's training transcripts (Rakib et al., 2023;
963,636 standard-Bangla sentences; 8.19 M tokens, 156,643 types), restricted to
types occurring ≥ 3 times (88,262) to exclude typos. A transcript word outside
this vocabulary is a *candidate regional word*. The same out-of-vocabulary
(OOV) construction is used by the Ben-10 paper to characterise its districts.

**(iii) Regional lexicons** (`build_lexicon.py external|indomain`). A lexicon
maps each region to {word → p(region | word)}, where p is the word's
size-normalised rate in that region divided by its summed rate over all
regions in the source (add-0.5 smoothing), so a word shared by several
dialects counts less. Two sources: the *external* lexicon from human-written
dialect text independent of the thesis audio — Ben-10 train transcripts
(district-labelled, CC0) and the regional side of the Vashantor parallel
corpus (a word also present on the pair's standard side is not regional) —
which covers Barishal, Chattogram, Sylhet, Noakhali and Mymensingh by exact
district-name match only; and the *in-domain* lexicon from the ASR transcripts
of the **train split only** (verified programmatically: no val/test clip
contributes), keeping a word if it occurs ≥ 3 times in ≥ 2 pseudo-speakers of
that class, which suppresses ASR noise and programme-specific words. In-domain
lexicons are built separately per front-end and per split protocol.

**(iv) Classifiers** (`lexical_did.py`). *Word matching*: each clip receives a
per-region *match percentage* — the weighted share of its tokens found in that
region's lexicon — and its share of non-formal tokens. A clip below a
non-formal-share threshold τ (tuned on val) or matching no lexicon is called
Formal; otherwise the best-matching region wins. No training; the per-word
evidence is shown by `predict.py --explain`. *TF-IDF + logistic regression*
over the normalised transcript (word 1–2-grams and character 2–5-grams,
sublinear TF, balanced class weights; C ∈ {0.3, 1, 3, 10, 30} chosen on val):
the learned text baseline, which uses formal words too. *Late fusion*:
α·log p_XLS-R + (1−α)·log p_TF-IDF with α ∈ [0, 1] in steps of 0.05 chosen on
val; XLS-R probabilities come from the Phase 2/3 checkpoints on the same 5 s
windows (`predict.py --dump-probs`; the argmax reproduces the saved test
predictions exactly on the strict split and on 1,328 of 1,329 clip-level
clips, the one difference being M1-vs-T4 arithmetic).

**(v) ASR reference evaluation** (`ben10_eval.py`). The thesis audio has no
reference text, so ASR quality is measured on Ben-10's public validation set
(1,666 clips, 10 districts, dialect transcribed as spoken). Besides corpus WER
and CER on normalised tokens, *dialect-word recall* is the share of reference
tokens outside the formal vocabulary that the hypothesis reproduces exactly —
the full-set, automated counterpart of the Ben-10 paper's hand-counted
"dialect recall" over ~50 words per district. Per-front-end OOV rates on the
thesis clips (`asr_diagnostics.py`) need no references and are reported
alongside.
