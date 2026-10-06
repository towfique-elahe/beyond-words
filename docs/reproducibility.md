# Reproducibility — exact recipe for every number

Phase 4 release document. Every result in the thesis/README maps to one command
below, run at seed 42 on the committed split files.

## Environment

- **Local (any platform / Apple Silicon):** Python 3.12, `pip install -r
  requirements.txt` in a venv. PyTorch auto-selects `cuda` > `mps` > `cpu`.
  MLP training uses `--device cpu` for bit-reproducibility.
- **CUDA/conda alternative:** `environment.yml`.
- **Cloud fine-tuning:** Kaggle T4 (`kaggle kernels push --accelerator
  NvidiaTeslaT4`; the default GPU assignment may be a P100, unsupported by
  current PyTorch builds). Dataset attached as a private Kaggle dataset.

## Canonical inputs (committed)

| File | Role |
|---|---|
| `splits/manifest_split_rel.csv` | clip-level split (75/10/15, seed 42): 6,977/997/1,329 |
| `splits/manifest_split_strict_rel.csv` | strict pseudo-speaker split: 6,931/1,045/1,327; 1,683 groups, 0 cross-split |

Relative paths resolve against the audio root (not distributed). Regenerating
from raw audio: `build_manifest.py --standardize` → `build_split.py` (canonical);
`build_speaker_clusters.py embed` + `cluster` → `build_split.py` (strict).
Note: clustering on different hardware may produce slightly different
pseudo-speakers; the committed strict split is the canonical one used everywhere.

## Result → command map

| Result (test macro-F1) | Command |
|---|---|
| Whisper clip-level **0.917** | `baseline_whisper.py cache` then `baseline_whisper.py train --device cpu` |
| Whisper strict **0.793** | `baseline_whisper.py train --device cpu --remap-split manifest_split_strict.csv` |
| MFCC clip-level **0.727** | `baseline_mfcc.py cache` then `baseline_whisper.py train --emb features_v1.npz --device cpu` |
| MFCC strict **0.480** | same + `--remap-split manifest_split_strict.csv` |
| XLS-R clip-level **0.952** | `finetune_xlsr.py --manifest splits/manifest_split_rel.csv` (T4, 20 ep) |
| XLS-R strict **0.808** | `finetune_xlsr.py --manifest splits/manifest_split_strict_rel.csv` (T4, 20 ep) |
| XLS-R strict + augment **0.748** | same + `--augment` |
| Data-efficiency table/figure | `data_efficiency.py` (3 seeds × 5 fractions × 2 reps × 2 splits) |
| Bootstrap 95 % CIs (all scores + gaps) | `bootstrap_ci.py` → `reports/phase3/bootstrap_ci.json` |

### Phase 5 (lexical route)

External text inputs (not committed; `data/` is gitignored): OOD-Speech
`train.csv` (Kaggle competition `bengaliai-speech`, rules must be accepted on the
account), Ben-10 `train/*/train.csv` + `valid/valid.csv` (HF `bengaliAI/Ben-10`,
CC0), Vashantor CSVs (Mendeley `bj5jgk878b` v2, CC BY 4.0).

| Result | Command |
|---|---|
| Transcripts, 9,303 clips × 5 ASR front-ends | `notebooks/kaggle_transcribe.ipynb` (T4; wraps `transcribe.py`; ~43 / 45 / 82 / 15 / 7 min for Tugstugi-Ben10 / Tugstugi / Whisper-large-v3 / wav2vec2 / Hishab) → `transcripts/<tag>.csv` |
| Formal vocabulary (156,643 types) | `build_lexicon.py formal --source data/ood_speech/train.csv` → `lexicons/formal_vocab.txt` (count ≥ 3 applied at load time) |
| External regional lexicon | `build_lexicon.py external --ben10 data/ben10/train.csv --vashantor data/vashantor/Vashantor_CSV_Format` |
| In-domain lexicon per front-end × split | `build_lexicon.py indomain --transcripts transcripts/<tag>.csv --manifest splits/<split>.csv --out lexicons/indomain_<tag>_<strict|clip>.json` (min-count 3, ≥ 2 pseudo-speakers) |
| XLS-R val/test probabilities for fusion | `predict.py --checkpoint checkpoints/xlsr_strict_best.pt --manifest splits/manifest_split_strict_rel.csv --audio-root <root> --dump-probs reports/phase5/xlsr_probs_strict.csv` (and `xlsr_best.pt` / `manifest_split_rel.csv` → `xlsr_probs_clip.csv`); argmax reproduces the saved predictions (strict 100 %, clip-level 1,328/1,329) |
| Word match / TF-IDF / fusion, e.g. Ben-10 strict **0.360 / 0.633 / 0.856** | `lexical_did.py --transcripts transcripts/tugstugi_ben10.csv --manifest splits/manifest_split_strict_rel.csv --lexicon lexicons/indomain_tugstugi_ben10_strict.json --xlsr-probs reports/phase5/xlsr_probs_strict.csv` → `reports/phase5/metrics_*.json`, `test_predictions_*.csv` |
| Front-end OOV diagnostics | `asr_diagnostics.py transcripts/<tag>.csv --min-count 3 --out reports/phase5/diagnostics_<tag>.json` |
| Summary table | `summarize_phase5.py` → `reports/phase5/summary.csv` |
| Bootstrap CIs incl. lexical runs | `bootstrap_ci.py --out reports/phase5/bootstrap_ci.json --extra "strict:Fusion/tugstugi_ben10=reports/phase5/test_predictions_tugstugi_ben10__indomain_tugstugi_ben10_strict__fusion.csv" ...` |
| WER / CER / dialect-word recall on Ben-10 valid | `notebooks/kaggle_ben10_eval.ipynb` (T4) → `transcripts/ben10_valid/<tag>.csv`, then `ben10_eval.py transcripts/ben10_valid/*.csv` |

Decoding is greedy everywhere (Whisper `num_beams=1`, `max_new_tokens=128`,
language/task forced to Bangla transcription; CTC argmax without a language
model). Thresholds tuned on val only: `tau` (Formal cut-off) for word matching,
`C` for TF-IDF, `alpha` for fusion. T4 fp16 and M1 fp32 Whisper transcripts agree
on 93.6 % of pilot clips; the committed numbers use the T4 transcripts.

All fine-tuning defaults are in `finetune_xlsr.py` argparse (batch 8 × accum 2,
enc lr 2e-5 / head lr 1e-4, 10 % warmup, fp16, frozen conv encoder, clip 1.0).
The Kaggle notebook `notebooks/kaggle_xlsr_finetune.ipynb` wraps the two GPU rows
(`MANIFEST` and `EXTRA_ARGS` variables select split and augmentation).

## Determinism notes

- MLP-head results are bit-reproducible on CPU at seed 42. MPS/CUDA runs land
  within ±0.005 macro-F1.
- Fine-tuning on T4/fp16 is reproducible at the reported precision at seed 42
  (identical data order; cuDNN nondeterminism affects the 4th decimal).
- Whisper/ECAPA/MFCC caches are deterministic given the manifest row order.

## Artifacts inventory

| Artifact | Where | Tracked? |
|---|---|---|
| Metrics, predictions, eval reports | `reports/phase2/`, `reports/phase3/`, `reports/phase5/` | yes |
| ASR transcripts, lexicons | `transcripts/`, `lexicons/` | small CSV/JSON — recommended to track (transcripts are derived from non-distributed audio but contain no audio) |
| External text corpora (OOD-Speech, Ben-10, Vashantor) | `data/` | no — download per the licences above |
| Figures | `reports/figures/` | yes |
| Split files | `splits/` | yes |
| Embedding/feature caches (`*.npz`) | repo root | no — regenerate |
| Checkpoints `xlsr_best.pt`, `xlsr_strict_best.pt` (1.2 GB each) | `checkpoints/` | no — regenerate on Kaggle or request from author |
| Raw audio | private | no — not redistributable |

## Release checklist (Phase 4)

- [ ] Tag a release (`v2.0`) once thesis numbers are frozen
- [ ] Verify a clean-clone dry run: pip install → baseline pipeline on a few clips
- [ ] Decide checkpoint hosting (HF Hub private/public, or on request)
- [ ] Optional: inference demo script (`predict.py <wav>` → dialect + confidence,
      using `checkpoints/xlsr_strict_best.pt`)
- [ ] Archive the exact Kaggle notebook versions used (v3 clip-level, strict v1,
      augment v1) — they are pinned on kaggle.com under the author's account
