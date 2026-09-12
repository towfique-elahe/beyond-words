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
| **macro** | **0.728** | **0.917** | **0.952** |

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
| v1 MFCC stats + MLP | 0.728 | 0.480 | −24.8 |
| Frozen Whisper-small + MLP | 0.917 | 0.793 | −12.4 |
| Fine-tuned XLS-R-300M | 0.952 | 0.808 | −14.4 |
| XLS-R-300M + channel augmentation | — | 0.748 | — |

Figure: `ablation_representation_split.png`.

Key results:
1. **Leakage is measured at 12–14 macro-F1 points** for the pretrained models
   (24.8 for MFCC). Strict numbers (0.793 / 0.808) are the honest estimates;
   clip-level numbers remain reported for comparability with prior work.
2. **Formal barely moves** (0.970 → 0.929 baseline): its split was source-disjoint
   under both protocols, so its stability isolates the protocol change as the
   cause of the regional drops.
3. **Fine-tuning still wins under strict evaluation** (+1.5 over frozen), but most
   of its clip-level margin (+3.5) came from channel exploitation — its strict
   drop (−14.4) slightly exceeds the frozen baseline's (−12.4).
4. Per-class strict numbers are noisier than the macro average: some test cells
   are dominated by a single large pseudo-speaker cluster (Barishal's strict test
   set is essentially one 251-clip cluster; its strict F1 of 0.60 baseline / 0.51
   XLS-R should be read with that caveat).

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
