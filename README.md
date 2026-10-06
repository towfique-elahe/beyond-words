# Beyond Words (v2)

**Bangla Dialect Identification with Self-Supervised Speech Models**

Classifying regional Bangla accents from short audio clips. v2 replaces the original hand-crafted-features + MLP pipeline with pretrained self-supervised speech models and a leakage-aware data pipeline. Headline results: **0.917** test macro-F1 with frozen Whisper-small embeddings, **0.952** after end-to-end XLS-R-300M fine-tuning, and — under a strict pseudo-speaker split that removes speaker/channel leakage — honest estimates of **0.793 / 0.808**.

> Thesis: *A Neural Network Approach to Classifying Bangla Accentual Diversity* — framed as a Dialect Identification (DID) task.

---

## Dialect classes (9)

| Class | Clips |
|---|---|
| Barishal | 1,257 |
| Chattogram | 797 |
| Dhaka | 763 |
| Formal (standard register) | 1,652 |
| Khulna | 750 |
| Mymensingh | 1,053 |
| Noakhali | 1,213 |
| Rajshahi | 860 |
| Sylhet | 958 |
| **Total** | **9,303** (~13.1 hours; 2 duplicate `.ogg` files in Formal are excluded) |

Clips are `.wav`, standardized to 16 kHz mono PCM-16, and nominally 5 s long (median 5.0 s; 1,067 fall outside 4.5–5.5 s, range 1.1–8.0 s). Audio was collected from TV shows, films, and online content, originally in mp4.

**Note on "Formal":** Formal is a *speech register* (standard/prescribed Bangla), not a regional category. Treating it as a ninth class is a deliberate design choice, discussed in the thesis.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Features | Hand-crafted (MFCC etc.) in `features.csv` | Raw audio → pretrained speech-model embeddings |
| Model | Small feed-forward NN | Whisper-small encoder + weighted MLP head (baseline); fine-tuned XLS-R-300M |
| Splitting | Random | Stratified + source-disjoint where metadata exists; **strict pseudo-speaker split** (ECAPA clustering) for honest evaluation |
| Imbalance | Unhandled | Inverse-frequency class weights; macro-F1 reported |
| Leakage awareness | None | Source IDs tracked; limitations disclosed (see below) |

## Pipeline

```
build_manifest.py  →  manifest.csv          # scan, validate, standardize audio, extract source_id
build_split.py     →  manifest_split.csv    # train/val/test (group-aware where possible)
baseline_whisper.py cache  →  embeddings.npz  # Whisper-small encoder, mean-pooled, cached once
baseline_whisper.py train                     # weighted MLP head + evaluation
finetune_xlsr.py           →  xlsr_best.pt    # Phase 2: end-to-end XLS-R-300M (Kaggle T4)
build_speaker_clusters.py  →  manifest_speaker.csv       # Phase 3: ECAPA pseudo-speakers
build_split.py --manifest manifest_speaker.csv           #   → strict group-disjoint split
baseline_mfcc.py cache     →  features_v1.npz # Phase 3: v1-style MFCC ablation row
transcribe.py              →  transcripts/<asr>.csv   # Phase 5: off-the-shelf Bangla ASR per clip (Kaggle T4)
build_lexicon.py formal|external|indomain →  lexicons/  # Phase 5: standard vocab + regional word lists
lexical_did.py             →  reports/phase5/ # Phase 5: word matching, TF-IDF, fusion with XLS-R
```

### Setup

**macOS (Apple Silicon) or any pip-based setup** — PyTorch uses the MPS backend on Apple GPUs automatically:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Linux/Windows with NVIDIA GPU** (conda):

```bash
conda env create -f environment.yml
conda activate beyond-words
```

### Run

```bash
# 1. Manifest (validates + converts to 16k mono in place)
python build_manifest.py --root "path/to/bangla_accent_voice_data" --out manifest.csv --standardize

# 2. Split (75/10/15 stratified by district; grouped by source_id where available)
python build_split.py --manifest manifest.csv --out manifest_split.csv

# 3. Baseline
python baseline_whisper.py cache --manifest manifest_split.csv --out embeddings.npz
python baseline_whisper.py train --emb embeddings.npz --epochs 60 --device cpu

# 4. Strict split + local Phase 3 evaluations
python build_speaker_clusters.py embed   --manifest manifest_split.csv --out ecapa_embeddings.npz
python build_speaker_clusters.py cluster --manifest manifest_split.csv --emb ecapa_embeddings.npz --out manifest_speaker.csv
python build_split.py --manifest manifest_speaker.csv --out manifest_split_strict.csv
python baseline_whisper.py train --emb embeddings.npz --device cpu --remap-split manifest_split_strict.csv
python baseline_mfcc.py cache --manifest manifest_split.csv --out features_v1.npz
python data_efficiency.py
python bootstrap_ci.py

# 5. Inference demo (any audio file/format; needs a trained checkpoint)
python predict.py path/to/clip.wav
```

### Phase 2: fine-tune XLS-R on Kaggle (free T4)

End-to-end fine-tuning of `facebook/wav2vec2-xls-r-300m` needs a CUDA GPU; a free
Kaggle T4 (16 GB, 30 GPU-h/week) covers it in one ~2 h session. The run uses the
**same split** as the baseline via [`splits/manifest_split_rel.csv`](splits/manifest_split_rel.csv)
(relative paths, committed for reproducibility), so results are directly comparable.

1. **Upload the dataset once (keep it private — the audio is not redistributable):**
   a `dataset-metadata.json` is expected in the audio root; edit the `id` to your
   Kaggle username, then:
   ```bash
   pip install kaggle          # auth: kaggle.json, or an access token in ~/.kaggle/access_token
   kaggle datasets create -p path/to/bangla_accent_voice_data --dir-mode zip
   ```
   (Kaggle datasets are private by default; or use the website's *Create Dataset* UI.)
2. **Run the notebook:** upload [`notebooks/kaggle_xlsr_finetune.ipynb`](notebooks/kaggle_xlsr_finetune.ipynb)
   to Kaggle, attach the dataset, set accelerator to GPU T4 and Internet ON, *Save & Run All*.
   (Or push it headlessly: `kaggle kernels push --accelerator NvidiaTeslaT4` with a
   `kernel-metadata.json` — forcing the T4 matters, as Kaggle's default GPU assignment
   can hand out a P100 that current PyTorch builds no longer support.) The notebook's
   `MANIFEST` / `EXTRA_ARGS` variables select the strict split and `--augment`.
3. **Collect outputs** from the notebook's Output tab: `xlsr_best.pt`,
   `metrics_xlsr.json`, `test_predictions_xlsr.csv`.

`finetune_xlsr.py` also runs locally (auto-selects `cuda` > `mps` > `cpu`) — use
`--limit 48 --epochs 1` for a smoke test; a full run on Apple Silicon is impractical.

## Baseline results (v2, frozen Whisper-small embeddings + MLP)

**Test macro-F1: 0.917** (accuracy 0.916, n = 1,329; seed 42)

| District | F1 |
|---|---|
| Khulna | 0.986 |
| Formal | 0.970 |
| Sylhet | 0.951 |
| Rajshahi | 0.947 |
| Dhaka | 0.913 |
| Mymensingh | 0.890 |
| Chattogram | 0.880 |
| Barishal | 0.871 |
| Noakhali | 0.848 |

Main confusion pair: **Barishal ↔ Noakhali** (southern coastal dialects), consistent with known dialectological proximity; Mymensingh, a transitional dialect zone, spreads its errors across several neighboring classes. Noakhali and Barishal are the hardest classes.

<p align="center">
  <img src="reports/figures/confusion_matrix_baseline.png" alt="Baseline confusion matrix" width="49%" />
  <img src="reports/figures/per_class_f1_baseline.png" alt="Baseline per-class F1" width="49%" />
</p>

t-SNE of the frozen Whisper-small embeddings, colored by dialect:

<p align="center">
  <img src="reports/figures/tsne_embeddings.png" alt="t-SNE of Whisper-small embeddings" width="70%" />
</p>

## Fine-tuned results (Phase 2, XLS-R-300M end-to-end)

**Test macro-F1: 0.952** (n = 1,329; 20 epochs, fp16, Kaggle T4; seed 42; same split as the baseline) — **+3.5 points over the frozen-embedding baseline**, with every district improving or holding. The largest gains land exactly on the baseline's weakest classes: Noakhali +6.1, Chattogram +4.4, Dhaka +5.9, Barishal +5.6.

| District | Baseline F1 | Fine-tuned F1 |
|---|---|---|
| Khulna | 0.986 | 0.986 |
| Sylhet | 0.951 | 0.982 |
| Rajshahi | 0.947 | 0.980 |
| Formal | 0.970 | 0.973 |
| Dhaka | 0.913 | 0.972 |
| Barishal | 0.871 | 0.927 |
| Chattogram | 0.880 | 0.924 |
| Mymensingh | 0.890 | 0.912 |
| Noakhali | 0.848 | 0.910 |
| **macro** | **0.917** | **0.952** |

The Barishal ↔ Noakhali confusion shrinks (31 → 15 errors) but remains the dominant error mode; Sylhet becomes near-perfect (137/137 recall). An 8-epoch run scored only 0.865 (still climbing); the 20-epoch validation curve plateaus at ~0.95 from epoch 11 onward, so longer training offers little further gain.

<p align="center">
  <img src="reports/figures/confusion_matrix_xlsr.png" alt="XLS-R confusion matrix" width="49%" />
  <img src="reports/figures/per_class_f1_comparison.png" alt="Per-class F1 comparison" width="49%" />
</p>

Run artifacts (metrics, test predictions) live in [`reports/phase2/`](reports/phase2/); the checkpoint (1.2 GB) is not tracked in git.

## Strict-split results (Phase 3: how much was leakage?)

The clip-level scores above are disclosed as an **upper bound** because regional
classes lack speaker metadata. Phase 3 measures the gap. `build_speaker_clusters.py`
embeds every clip with ECAPA-TDNN and clusters per district (agglomerative, cosine);
the distance threshold is **calibrated on the Formal class**, whose true YouTube
source IDs are known (0.50 maximizes adjusted Rand index against them). The
resulting 1,683 pseudo-speaker groups are verified never to cross splits, giving a
strict group-disjoint split ([`splits/manifest_split_strict_rel.csv`](splits/manifest_split_strict_rel.csv)).
All three representations were then re-evaluated under the identical protocol
(same MLP/weighted loss/seed; XLS-R retrained from scratch on the strict train set):

| Representation | Clip-level split | Strict split | Δ |
|---|---|---|---|
| v1 MFCC stats + MLP (`baseline_mfcc.py`) | 0.727 | 0.480 | −24.8 |
| Frozen Whisper-small + MLP | 0.917 | 0.793 | −12.4 |
| Fine-tuned XLS-R-300M | 0.952 | 0.808 | −14.4 |
| Fine-tuned XLS-R-300M + train-time augmentation | — | 0.748 | — |

<p align="center">
  <img src="reports/figures/ablation_representation_split.png" alt="Ablation: representation x split" width="70%" />
</p>

Findings:

- **Roughly 12–14 points of the headline scores are attributable to speaker/channel
  overlap**, now measured rather than speculated. Strict-split numbers are the
  honest estimates; clip-level numbers remain comparable to prior work.
- **Self-supervised representations are more leakage-robust than hand-crafted
  features**: MFCCs lose 24.8 points under the strict split — half their apparent
  skill was channel signature — versus 12–14 for the pretrained models.
- **Under the strict split, fine-tuned XLS-R and frozen Whisper are statistically
  indistinguishable**: XLS-R's +1.5-point lead has a speaker-group bootstrap 95 %
  CI of −2.0 to +7.5 (`bootstrap_ci.py`). Its clip-level lead (+3.5, CI +1.8 to
  +5.3) is significant, so most of that advantage came from exploiting
  recording-channel cues.
- **Uncertainty is large under the strict split.** Only 230 pseudo-speaker groups
  make up the strict test set, and Barishal's is a *single* 251-clip cluster, so
  group-bootstrap 95 % CIs span ~12 points (Whisper 0.688–0.808, XLS-R
  0.717–0.838). The MFCC-vs-SSL gaps and the augmentation drop remain significant
  under this stricter test; the XLS-R-vs-Whisper gap does not.
- **Augmentation is a negative result.** Training with a channel-suppression
  chain (gain, additive noise, 7-band EQ, telephone band-pass, pitch ±2 st,
  tempo 0.9–1.1; `finetune_xlsr.py --augment`) *lowered* strict macro-F1 from
  0.808 to 0.748. The likely mechanism is that pitch and tempo perturbations
  destroy dialect-bearing prosodic cues — for DID, unlike ASR, prosody is signal,
  not nuisance. A milder, prosody-preserving chain is future work.
- Caveats: pseudo-speakers are approximate (calibration ARI 0.347), and some strict
  test cells are dominated by single large clusters (e.g. Barishal), making
  per-class strict numbers noisier than the macro average.

### Data efficiency

`data_efficiency.py` retrains the identical MLP head on stratified fractions of
the train split (3 seeds per point, min–max bands):

<p align="center">
  <img src="reports/figures/data_efficiency.png" alt="Data-efficiency curves" width="70%" />
</p>

- **Frozen Whisper embeddings with 10 % of the labels (0.79 clip-level / 0.67
  strict) outperform MFCCs with 100 % (0.73 / 0.49)** — pretraining is worth more
  than a tenfold increase in labeled data here.
- Curves flatten beyond ~50 % of the data: the ~13 h corpus is adequately sized
  for the frozen-embedding approach.
- The representation gap widens under the strict split at every fraction — the
  leakage-robustness advantage of self-supervised features holds across scales.

Artifacts in [`reports/phase3/`](reports/phase3/): strict-split evaluations for all
three models, XLS-R strict metrics/predictions, `strict_split_results.json`, and
`bootstrap_ci.json` (clip- and speaker-group bootstrap CIs for every score and gap).

## Lexical route (Phase 5: is the dialect in the words or in the sound?)

Phases 1–3 never look at *what* is said: the models consume log-mel frames or
the raw waveform. Phase 5 adds a text route. Every clip is transcribed with
off-the-shelf Bangla ASR (no fine-tuning), words outside a standard-Bangla
vocabulary are treated as candidate regional words, and three classifiers run
on the transcripts under the same splits and the same macro-F1 protocol:

- **Word matching** — per-region *match percentage*: the weighted share of a
  clip's words found in that region's lexicon (`build_lexicon.py indomain`,
  built from train-split transcripts only; `external` from Ben-10 + Vashantor
  human text). Interpretable, no training.
- **TF-IDF + logistic regression** on the transcripts (word 1–2 grams, char 2–5 grams).
- **Late fusion** of the TF-IDF probabilities with XLS-R (`alpha` tuned on val).

The standard-Bangla vocabulary is OOD-Speech `train.csv` (156k word types; words
with count ≥ 3 are used). Transcripts come from five front-ends, chosen to
follow the Ben-10 ASR benchmark (IJCNLP-AACL 2025): `bengaliAI/tugstugi_bengaliai-regional-asr_whisper-medium`
(fine-tuned on Ben-10 dialect speech), `bengaliAI/tugstugi_bengaliai-asr_whisper-medium`,
`openai/whisper-large-v3`, `hishab/hishab_bn_fastconformer`, and
`arijitx/wav2vec2-xls-r-300m-bengali` as a stand-in for the paper's unreleased
"Wav2Vec2 (SCB)". Google ASR (paid) and the paper's Ben-10 wav2vec2 (no public
checkpoint) are not run.

Test macro-F1 (`reports/phase5/summary.csv`; XLS-R on the same split: 0.952 clip-level, 0.808 strict):

| ASR front-end | OOV share: Formal / regional | Word match (strict) | TF-IDF (strict) | Fusion (strict) | TF-IDF (clip) | Fusion (clip) |
|---|---|---|---|---|---|---|
| Tugstugi (Ben-10) | 0.08 / 0.25 | 0.360 (44 % of clips matched) | **0.633** | **0.856** | **0.728** | **0.967** |
| Tugstugi | 0.05 / 0.13 | 0.060 | 0.456 | 0.836 | 0.595 | 0.965 |
| Hishab FastConformer | 0.04 / 0.09 | 0.049 | 0.445 | 0.820 | 0.553 | 0.962 |
| wav2vec2 XLS-R Bengali | 0.27 / 0.51 | 0.077 | 0.424 | 0.825 | 0.536 | 0.960 |
| Whisper-large-v3 | 0.45 / 0.53 | 0.156 | 0.388 | 0.800 | 0.490 | 0.954 |

What this says:

- **Standard-Bangla ASR erases the dialect.** Hishab and base Tugstugi output
  as few out-of-vocabulary words on regional clips as on Formal ones; only the
  Ben-10-tuned model keeps dialect forms (মুই, আঁই, হামার, কিতা …), and it is the
  only front-end on which word matching works at all. Whisper-large-v3 and the
  wav2vec2 model have *high* OOV shares for the wrong reason — misrecognitions
  and repetition loops (12 % of large-v3 transcripts repeat a word 5+ times).
- **Matching regional words alone is weak** (0.36 strict): fewer than half the
  5-second test clips contain any word from the lexicon. Using all words and
  character patterns (TF-IDF) nearly doubles the score — the formal-word filter
  throws away evidence such as dialect verb endings.
- **Text and sound are complementary, modestly.** Fusion improves on XLS-R
  with four of the five front-ends (Whisper-large-v3 is the exception on the
  strict split). With the Ben-10 model the gain is +4.8 points strict (95 %
  speaker-group CI −0.3 to +6.8 — borderline, 4 % probability of no gain) and
  +1.5 clip-level (CI +0.9 to +2.4). No other front-end's strict gain is
  significant; clip-level gains are significant for all but Whisper-large-v3.
- **The lexical route alone sits between MFCC and frozen Whisper** (0.63 strict
  vs 0.48 / 0.79): the words carry real dialect information, but less than the
  acoustics on clips this short.

**ASR quality against human references** (Ben-10 valid, 1,666 clips, 10 districts;
`ben10_eval.py`, greedy decoding, punctuation ignored). *Dialect recall* is the
share of reference words outside the standard vocabulary (21,119 tokens) that
the hypothesis reproduces exactly — an automated, full-set version of the
Ben-10 paper's hand-counted measure:

| Front-end | WER | CER | Dialect-word recall | Formal-word recall |
|---|---|---|---|---|
| Tugstugi (Ben-10) | **0.707** | **0.471** | **0.192** | **0.455** |
| Tugstugi | 0.775 | 0.532 | 0.034 | 0.369 |
| Hishab FastConformer | 0.792 | 0.517 | 0.012 | 0.405 |
| wav2vec2 XLS-R Bengali | 0.942 | 0.669 | 0.012 | 0.085 |
| Whisper-large-v3 | 0.960 | 0.808 | 0.016 | 0.061 |

The WER ordering and magnitudes match the paper's Table 3 (0.70 for the Ben-10
model on its private test set). Even the best front-end recovers fewer than one
in five dialect words (best district Sylhet 0.24, worst Rangpur 0.13), which
bounds what any word-matching approach can see.

`python predict.py --explain clip.wav` adds the transcript, the words outside
standard Bangla and the per-region match percentages to the acoustic top-k.
Artifacts in [`reports/phase5/`](reports/phase5/): per-front-end diagnostics
(`diagnostics_*.json`), per-run metrics and test predictions, `bootstrap_ci.json`,
`summary.csv`. Transcripts (`transcripts/`) and lexicons (`lexicons/`) are
regenerated with the pipeline above; ASR quality against human references is
measured on Ben-10 valid with `ben10_eval.py`.

## Known limitations (disclosed by design)

- **Source/channel leakage in the canonical split — now measured.** About a third of Formal clips (557 of 1,652, 35 sources) carry recoverable YouTube video IDs and are split source-disjoint. All other clips have no recoverable speaker/source metadata, so the canonical split is clip-level for them and the same speaker or recording source may appear in both train and test. **Clip-level scores are therefore an upper bound** — quantified by the strict-split results above at 12–14 macro-F1 points for the pretrained models. Near-perfect clip-level scores for individual classes (e.g., Khulna) partly reflect recording-channel signatures.
- **Class imbalance** (Formal 1,652 vs. Khulna 750) is mitigated with weighted loss; macro-F1 is the headline metric, not accuracy.
- Clips are ~5 s; longer-context dialect cues are not modeled.

Mitigation status: speaker clustering (ECAPA) is implemented — the strict-split results above quantify the leakage. Channel-suppression augmentation was tested and *reduced* strict-split accuracy (negative result above); prosody-preserving augmentation remains future work.

## Roadmap

- [x] Phase 0 — data hygiene, manifest, leakage-aware splitting
- [x] Phase 1 — cached Whisper-small embedding baseline
- [x] Phase 2 — fine-tune `facebook/wav2vec2-xls-r-300m` end-to-end (fp16, Kaggle T4) — **0.952 test macro-F1**
- [x] Phase 3 — speaker-clustered strict split (**0.793 / 0.808** honest estimates), ablation table, augmentation study (negative result), data-efficiency curve
- [x] Phase 4 — thesis write-up drafts, reproducible release, inference demo (`predict.py`)
- [x] Phase 5 — lexical route: ASR transcripts, regional-word matching, TF-IDF, fusion (**0.856** strict with XLS-R + text); five ASR front-ends compared

## Hardware

The project runs on a zero-hardware budget. Phases 0–1 were developed on an **NVIDIA GTX 1660 Ti (6 GB)** and reproduced within 0.3 macro-F1 points on an **Apple M1 (16 GB)** via PyTorch's MPS backend — the device is auto-selected (`cuda` > `mps` > `cpu`). Everything except end-to-end fine-tuning (embedding caches, MLP heads, ECAPA clustering, ablations) runs locally on the M1; the XLS-R fine-tuning runs (Phases 2–3) each used ~2.5 h of Kaggle's free T4 tier (16 GB, 30 GPU-h/week).

## Repository layout

```
├── build_manifest.py      # Phase 0: scan + validate + standardize audio
├── build_split.py         # Phase 0: leakage-aware train/val/test split
├── baseline_whisper.py    # Phase 1: cache embeddings, train + evaluate head
├── finetune_xlsr.py       # Phase 2: end-to-end XLS-R-300M fine-tuning
├── build_speaker_clusters.py  # Phase 3: ECAPA pseudo-speaker clustering
├── baseline_mfcc.py       # Phase 3: v1-style MFCC features (ablation)
├── data_efficiency.py     # Phase 3: label-fraction sweep + curve
├── bootstrap_ci.py        # Phase 3: bootstrap CIs (clip + speaker-group) for scores and gaps
├── predict.py             # Phase 4: inference demo (dialect + confidence; --explain shows the words)
├── transcribe.py          # Phase 5: off-the-shelf Bangla ASR over the manifest (whisper / ctc / nemo)
├── text_utils.py          # Phase 5: shared Bangla normaliser + tokeniser
├── build_lexicon.py       # Phase 5: formal vocabulary, external + in-domain regional lexicons
├── lexical_did.py         # Phase 5: word matching, TF-IDF, late fusion with XLS-R
├── asr_diagnostics.py     # Phase 5: per-front-end OOV / empty-rate table (no references needed)
├── ben10_eval.py          # Phase 5: WER/CER + dialect-word recall on Ben-10 valid
├── summarize_phase5.py    # Phase 5: one table over front-ends x splits
├── splits/                # committed splits (relative paths, reproducibility)
│   ├── manifest_split_rel.csv         # canonical clip-level split
│   └── manifest_split_strict_rel.csv  # Phase 3 strict pseudo-speaker split
├── environment.yml        # conda environment (CUDA machines)
├── requirements.txt       # pip environment (macOS / Apple Silicon or any platform)
├── notebooks/             # analysis notebooks (EDA, figures)
│   ├── beyond-words.ipynb
│   ├── kaggle_xlsr_finetune.ipynb
│   ├── kaggle_transcribe.ipynb        # Phase 5: all clips x all ASR front-ends (T4)
│   └── kaggle_ben10_eval.ipynb        # Phase 5: front-ends on Ben-10 valid audio
├── reports/
│   ├── figures/           # result figures (baseline + fine-tuned + ablation)
│   ├── phase2/            # XLS-R run metrics + test predictions
│   ├── phase3/            # strict-split evaluations + ablation artifacts
│   └── phase5/            # lexical-route metrics, predictions, diagnostics, summary
├── docs/                  # thesis materials
│   ├── thesis/            #   chapter outline + methodology/results/discussion drafts
│   ├── reproducibility.md #   exact result→command map, release checklist
│   └── caveat.txt         #   original leakage caveat note
├── LICENSE                # MIT
└── README.md
```

Generated artifacts (manifests, `embeddings.npz`, `ecapa_embeddings.npz`,
`features_v1.npz`, model checkpoints) and the audio dataset are **not** tracked in
git — regenerate them with the pipeline above.

## Acknowledgements

Built on [OpenAI Whisper](https://github.com/openai/whisper) encoders, [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m) fine-tuning via the Hugging Face `transformers` ecosystem, [SpeechBrain](https://speechbrain.github.io/)'s ECAPA-TDNN for speaker clustering, and [audiomentations](https://github.com/iver56/audiomentations) for the augmentation study.

## License

Released under the [MIT License](LICENSE). The dataset audio is not covered by this license and is not distributed in this repository.
