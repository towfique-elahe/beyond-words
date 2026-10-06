# Chapter Draft — Results

All numbers are test macro-F1 unless stated; artifacts in `reports/`.

## 5.1 Clip-level results (canonical split, upper bound)

**Frozen Whisper-small + MLP: 0.917** (accuracy 0.916, n = 1,329).
**Fine-tuned XLS-R-300M: 0.952** — +3.5 points, every class improving or holding.

| District | MFCC v1 | Frozen Whisper | Fine-tuned XLS-R |
|---|---|---|---|
| Khulna | | 0.986 | 0.986 |
| Sylhet | | 0.951 | 0.982 |
| Rajshahi | | 0.947 | 0.980 |
| Formal | | 0.970 | 0.973 |
| Dhaka | | 0.913 | 0.972 |
| Barishal | | 0.871 | 0.927 |
| Chattogram | | 0.880 | 0.924 |
| Mymensingh | | 0.890 | 0.912 |
| Noakhali | | 0.848 | 0.910 |
| **macro** | **0.727** | **0.917** | **0.952** |

*(Per-class MFCC numbers: `reports/phase3/mfcc_clip_eval.txt`.)*

Observations:
- Fine-tuning's gains concentrate on the baseline's weakest classes
  (Noakhali +6.1, Dhaka +5.9, Barishal +5.6, Chattogram +4.4) — task adaptation
  reallocates capacity toward the hard, acoustically close dialects.
- The dominant confusion, Barishal ↔ Noakhali (southern coastal dialects), halves
  under fine-tuning (31 → 15 test errors) but is not eliminated. Mymensingh — a
  transitional dialect zone — spreads its errors across neighbors. Errors pattern
  along dialect geography, not randomly.
- Training dynamics: 8-epoch pilot 0.865 (still climbing); 20-epoch validation
  plateaus ~0.95 from epoch 11 (fig. `training_curve_xlsr.png`) — converged.

Figures: `confusion_matrix_baseline.png`, `per_class_f1_baseline.png`,
`confusion_matrix_xlsr.png`, `per_class_f1_comparison.png`, `tsne_embeddings.png`.

## 5.2 Strict-split results: quantifying leakage

The six-cell ablation (representation × split), plus the augmentation variant:

| Representation | Clip-level | Strict | Δ (points) |
|---|---|---|---|
| v1 MFCC stats + MLP | 0.727 | 0.480 | −24.8 |
| Frozen Whisper-small + MLP | 0.917 | 0.793 | −12.4 |
| Fine-tuned XLS-R-300M | 0.952 | 0.808 | −14.4 |
| XLS-R-300M + channel augmentation | — | 0.748 | — |

Figure: `ablation_representation_split.png`.

Key results:
1. **Leakage is measured at 12–14 macro-F1 points** for the pretrained models
   (24.8 for MFCC). Strict numbers (0.793 / 0.808) are the honest estimates;
   clip-level numbers remain reported for comparability with prior work.
2. **Under strict evaluation, fine-tuned and frozen are statistically tied.**
   XLS-R's +1.5 lead over frozen Whisper has a speaker-group bootstrap 95 % CI of
   −2.0 to +7.5; its clip-level +3.5 lead (CI +1.8 to +5.3) is significant. Most
   of the fine-tuning margin therefore came from channel exploitation — its
   strict drop (−14.4) slightly exceeds the frozen baseline's (−12.4).
3. Only the 557 sourced Formal clips were source-disjoint under *both*
   protocols; Formal's other 1,095 clips moved from clip-level to pseudo-speaker
   grouping like the regional classes, so Formal is **not** a clean control.
4. Per-class strict numbers are noisier than the macro average: some test cells
   are dominated by a single large pseudo-speaker cluster (Barishal's strict test
   set is essentially one 251-clip cluster; its strict F1 of 0.60 baseline / 0.51
   XLS-R should be read with that caveat).

### Statistical uncertainty (`bootstrap_ci.py`, B = 2000)

| Model | Clip-level (95 % CI, group bootstrap) | Strict (95 % CI, group bootstrap) |
|---|---|---|
| MFCC + MLP | 0.727 (0.676–0.753) | 0.480 (0.368–0.552) |
| Frozen Whisper + MLP | 0.917 (0.894–0.932) | 0.793 (0.688–0.808) |
| Fine-tuned XLS-R | 0.952 (0.935–0.963) | 0.808 (0.717–0.838) |
| XLS-R + augmentation | — | 0.748 (0.649–0.792) |

Groups are pseudo-speakers or YouTube sources (565 in the clip-level test set,
230 in the strict one). Clip-level bootstrap intervals are narrower but understate
the variance, because clips from one speaker are correlated. Significant under the
group bootstrap: every MFCC-vs-SSL gap, XLS-R vs Whisper at clip level, and the
augmentation drop (−6.0, CI −9.4 to −0.9). Not significant: XLS-R vs Whisper under
the strict split.

## 5.3 Data efficiency

Mean test macro-F1 over 3 seeds (full table: `reports/phase3/data_efficiency.csv`;
figure: `data_efficiency.png`):

| Fraction of train | 10 % | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|---|
| Whisper, clip-level | 0.788 | 0.854 | 0.902 | 0.908 | 0.913 |
| Whisper, strict | 0.671 | 0.751 | 0.784 | 0.790 | 0.793 |
| MFCC, clip-level | 0.522 | 0.618 | 0.675 | 0.707 | 0.726 |
| MFCC, strict | 0.371 | 0.434 | 0.475 | 0.489 | 0.494 |

1. **Whisper embeddings with 10 % of labels beat MFCC with 100 % under both
   protocols** — self-supervised pretraining substitutes for more than a tenfold
   increase in labeled data on this task.
2. Whisper curves flatten beyond ~50 %: the ~13 h corpus is adequately sized for
   the frozen-embedding approach; further collection would buy little there.
3. The representation gap widens under the strict split at every fraction — the
   leakage-robustness advantage of SSL features holds across data scales.

## 5.4 Augmentation study (negative result)

Training the strict-split XLS-R with the channel-suppression chain **reduced**
macro-F1 from 0.808 to 0.748 (validation depressed throughout training — best
epoch 17/20, val 0.702 vs. 0.757 unaugmented — so this is not selection noise).
Interpretation is taken up in the Discussion: pitch and tempo perturbations,
standard in ASR augmentation, plausibly destroy dialect-bearing prosodic cues.
Artifacts: `reports/phase3/metrics_xlsr_augment.json`.

## 5.5 The lexical route: words versus sound

Artifacts: `reports/phase5/` (`summary.csv`, `metrics_*.json`,
`test_predictions_*.csv`, `diagnostics_*.json`, `ben10_valid_eval.json`,
`bootstrap_ci.json`).

### 5.5.1 How much dialect survives transcription?

Ben-10 valid (1,666 human-transcribed clips; 21,119 reference tokens outside
the formal vocabulary):

| Front-end | WER | CER | Dialect-word recall | Formal-word recall |
|---|---|---|---|---|
| Tugstugi, fine-tuned on Ben-10 | **0.707** | **0.471** | **0.192** | **0.455** |
| Tugstugi (standard Bangla) | 0.775 | 0.532 | 0.034 | 0.369 |
| Hishab FastConformer | 0.792 | 0.517 | 0.012 | 0.405 |
| wav2vec2 XLS-R Bengali (SCB stand-in) | 0.942 | 0.669 | 0.012 | 0.085 |
| Whisper-large-v3 | 0.960 | 0.808 | 0.016 | 0.061 |

The WER ordering and magnitudes match the Ben-10 paper's Table 3 (0.70 for the
Ben-10 model on the private test set; 0.81 / 0.87 / 1.13 for Tugstugi / Hishab /
Whisper-large-v3). Even the best front-end reproduces fewer than one dialect
word in five (per district: Sylhet 0.24 … Rangpur 0.13); the standard-Bangla
models reproduce 1–3 %. The paper's hand-counted recall (0.16–0.53 vs
0.01–0.09 for the same two Tugstugi models) is confirmed at full scale and
automatically.

On the thesis clips (no references), the share of transcript tokens outside
the formal vocabulary, Formal class vs mean over the eight regional classes:
Ben-10 model 0.08 / 0.25; Tugstugi 0.05 / 0.13; Hishab 0.04 / 0.09;
wav2vec2 0.27 / 0.51; Whisper-large-v3 0.45 / 0.53. Only the Ben-10 model
separates regional from Formal speech by a wide margin with recognisable
dialect forms (মুই, আঁই, আঁর, হামার, কিতা, টেহা); the two high-OOV models are
high for the wrong reason — misrecognition, and for Whisper-large-v3
repetition loops in 12.3 % of transcripts (a word repeated five or more times
in a row) plus 2–5 % empty outputs per class.

### 5.5.2 Classification from transcripts

Test macro-F1 by front-end (in-domain lexicon; XLS-R on the same split: 0.808
strict, 0.952 clip-level):

| Front-end | Word match, strict | TF-IDF, strict | Fusion, strict | TF-IDF, clip | Fusion, clip |
|---|---|---|---|---|---|
| Tugstugi (Ben-10) | 0.360 (44 % of clips matched) | **0.633** | **0.856** | **0.728** | **0.967** |
| Tugstugi | 0.060 | 0.456 | 0.836 | 0.595 | 0.965 |
| Hishab FastConformer | 0.049 | 0.445 | 0.820 | 0.553 | 0.962 |
| wav2vec2 XLS-R Bengali | 0.077 | 0.424 | 0.825 | 0.536 | 0.960 |
| Whisper-large-v3 | 0.156 | 0.388 | 0.800 | 0.490 | 0.954 |

1. **Word matching is weak** even with the best front-end: 0.360 strict /
   0.399 clip-level. Fewer than half of the test clips contain any lexicon
   word, so the rest default to Formal. Relaxing the lexicon thresholds
   (min-count 1, any speaker count; chosen on val) raises coverage to 59 % and
   macro-F1 to 0.419 — still far below every pretrained acoustic model. The
   clean external lexicon (Ben-10 + Vashantor; 6 of 9 classes) scores 0.418 on
   its own classes with 51 % coverage. With the standard-Bangla front-ends,
   matching collapses (coverage 2–5 %): there are no regional words left to
   match.
2. **The learned text classifier roughly doubles the matching score** (0.633
   strict with the Ben-10 model), placing words alone between MFCC (0.480) and
   frozen Whisper (0.793). Front-end quality orders the TF-IDF results exactly
   as dialect-word recall does.
3. **Fusion improves on XLS-R with four of five front-ends.** With the Ben-10
   model: strict 0.808 → 0.856 (α = 0.25, i.e. the text carries three quarters
   of the log-probability weight after tuning on val); clip-level 0.952 → 0.967
   (α = 0.30).

### 5.5.3 Statistical uncertainty (`bootstrap_ci.py --extra …`, B = 2000)

| Model (Ben-10 front-end) | Clip-level (95 % CI, group) | Strict (95 % CI, group) |
|---|---|---|
| Word match | 0.399 (0.359–0.421) | 0.360 (0.284–0.380) |
| TF-IDF | 0.728 (0.684–0.748) | 0.633 (0.538–0.642) |
| Fusion XLS-R + TF-IDF | 0.967 (0.952–0.976) | 0.856 (0.754–0.875) |
| Fine-tuned XLS-R (reference) | 0.952 (0.935–0.963) | 0.808 (0.717–0.838) |

Gaps against XLS-R (fusion − XLS-R): clip-level +1.5 points, group CI +0.9 to
+2.4, P(gap ≤ 0) = 0.000 — significant; strict +4.8 points, group CI −0.3 to
+6.8, P(gap ≤ 0) = 0.043 — borderline. For the other front-ends, clip-level
fusion gains are significant except Whisper-large-v3 (+0.2, CI −0.7 to +1.1);
no other strict gain is significant, and Whisper-large-v3's fusion is 0.8
points *below* XLS-R on the strict split.

### 5.5.4 Where the text helps (strict split, Ben-10 front-end, per-class F1)

| Class | Word match | TF-IDF | XLS-R | Fusion | Δ fusion − XLS-R |
|---|---|---|---|---|---|
| Barishal | 0.268 | 0.544 | 0.508 | 0.720 | **+21.2** |
| Sylhet | 0.600 | 0.750 | 0.750 | 0.832 | +8.2 |
| Chattogram | 0.392 | 0.617 | 0.820 | 0.878 | +5.8 |
| Formal | 0.448 | 0.805 | 0.785 | 0.835 | +5.0 |
| Noakhali | 0.500 | 0.707 | 0.862 | 0.898 | +3.5 |
| Mymensingh | 0.395 | 0.694 | 0.904 | 0.928 | +2.4 |
| Rajshahi | 0.175 | 0.550 | 0.842 | 0.857 | +1.5 |
| Dhaka | 0.158 | 0.470 | 0.848 | 0.853 | +0.5 |
| Khulna | 0.305 | 0.557 | 0.957 | 0.904 | −5.3 |

The gain concentrates on the classes the acoustic model finds hardest under
the strict split (Barishal — the single-cluster test cell — and Sylhet) and
on the classes with the most distinctive, best-transcribed vocabulary
(Sylhet's match F1 of 0.60 is the only strong matching result). Khulna, the
acoustic model's best class, is the one class fusion hurts. Formal is the only
class on which TF-IDF alone beats XLS-R: the register is lexically marked.
