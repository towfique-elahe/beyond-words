# Beyond Words (v2)

**Bangla Dialect Identification with Self-Supervised Speech Models**

Classifying regional Bangla accents from short audio clips. v2 replaces the original hand-crafted-features + MLP pipeline with pretrained speech-model embeddings and (upcoming) end-to-end fine-tuning, plus a leakage-aware data pipeline.

> Thesis: *A Neural Network Approach to Classifying Bangla Accentual Diversity* — framed as a Dialect Identification (DID) task.

---

## Dialect classes (9)

| Class | Clips |
|---|---|
| Barishal | 1,257 |
| Chattogram | 797 |
| Dhaka | 763 |
| Formal (standard register) | 1,654 |
| Khulna | 750 |
| Mymensingh | 1,053 |
| Noakhali | 1,213 |
| Rajshahi | 860 |
| Sylhet | 958 |
| **Total** | **9,305** (~13 hours) |

All clips are ~5-second `.wav`, standardized to 16 kHz mono PCM-16. Audio was collected from TV shows, films, and online content, originally in mp4.

**Note on "Formal":** Formal is a *speech register* (standard/prescribed Bangla), not a regional category. Treating it as a ninth class is a deliberate design choice, discussed in the thesis.

## What changed from v1

| | v1 | v2 |
|---|---|---|
| Features | Hand-crafted (MFCC etc.) in `features.csv` | Raw audio → pretrained speech-model embeddings |
| Model | Small feed-forward NN | Whisper-small encoder + weighted MLP head (baseline); XLS-R-300M fine-tuning (planned) |
| Splitting | Random | Stratified; **source-disjoint where source metadata exists** |
| Imbalance | Unhandled | Inverse-frequency class weights; macro-F1 reported |
| Leakage awareness | None | Source IDs tracked; limitations disclosed (see below) |

## Pipeline

```
build_manifest.py  →  manifest.csv          # scan, validate, standardize audio, extract source_id
build_split.py     →  manifest_split.csv    # train/val/test (group-aware where possible)
baseline_whisper.py cache  →  embeddings.npz  # Whisper-small encoder, mean-pooled, cached once
baseline_whisper.py train                     # weighted MLP head + evaluation
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

# 2. Split (85/10/15 stratified by district; grouped by source_id where available)
python build_split.py --manifest manifest.csv --out manifest_split.csv

# 3. Baseline
python baseline_whisper.py cache --manifest manifest_split.csv --out embeddings.npz
python baseline_whisper.py train --emb embeddings.npz --epochs 60
```

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

## Known limitations (disclosed by design)

- **Possible source/channel leakage for regional classes.** Formal-class filenames contain recoverable YouTube video IDs, enabling a source-disjoint split. The 8 regional classes have only sequential filenames with no recoverable speaker/source metadata, so their split is clip-level. The same speaker or recording source may therefore appear in both train and test, and **reported scores for regional classes should be read as an upper bound.** Near-perfect scores for individual classes (e.g., Khulna) may partly reflect recording-channel signatures rather than accent alone.
- **Class imbalance** (Formal 1,654 vs. Khulna 750) is mitigated with weighted loss; macro-F1 is the headline metric, not accuracy.
- Clips are ~5 s; longer-context dialect cues are not modeled.

Planned mitigations: speaker clustering (ECAPA embeddings) to build pseudo-speaker groups for a stricter split, and augmentation (noise, reverb, speed/pitch, codec round-trip) to suppress channel cues.

## Roadmap

- [x] Phase 0 — data hygiene, manifest, leakage-aware splitting
- [x] Phase 1 — cached Whisper-small embedding baseline
- [ ] Phase 2 — fine-tune `facebook/wav2vec2-xls-r-300m` end-to-end (fp16, Kaggle/Colab T4)
- [ ] Phase 3 — ablation table (v1 features vs. frozen embeddings vs. fine-tuned), augmentation study, speaker-clustered strict split, data-efficiency curve
- [ ] Phase 4 — thesis write-up, reproducible release, inference demo

## Hardware

Phases 0–1 were developed on a single **NVIDIA GTX 1660 Ti (6 GB)** and later reproduced on an **Apple M1 (16 GB unified memory)** via PyTorch's MPS backend — the device is auto-selected (`cuda` > `mps` > `cpu`). All configurations are sized for a small-memory budget: fp32, frozen feature encoder, small batches, and one-time embedding caching. Phase 2 end-to-end fine-tuning targets a free cloud GPU (Kaggle/Colab T4, 16 GB VRAM).

## Repository layout

```
├── build_manifest.py      # Phase 0: scan + validate + standardize audio
├── build_split.py         # Phase 0: leakage-aware train/val/test split
├── baseline_whisper.py    # Phase 1: cache embeddings, train + evaluate head
├── environment.yml        # conda environment (CUDA machines)
├── requirements.txt       # pip environment (macOS / Apple Silicon or any platform)
├── notebooks/             # analysis notebooks (EDA, figures)
│   └── beyond-words.ipynb
├── reports/figures/       # baseline result figures
├── docs/                  # thesis notes (leakage caveat, etc.)
├── LICENSE                # MIT
└── README.md
```

Generated artifacts (`manifest.csv`, `manifest_split.csv`, `embeddings.npz`) and the
audio dataset are **not** tracked in git — regenerate them with the pipeline above.

## Acknowledgements

Built on [OpenAI Whisper](https://github.com/openai/whisper) encoders and the Hugging Face `transformers` ecosystem. Planned fine-tuning uses [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m).

## License

Released under the [MIT License](LICENSE). The dataset audio is not covered by this license and is not distributed in this repository.
