# Thesis Outline — Beyond Words

*A Neural Network Approach to Classifying Bangla Accentual Diversity*
(framed as Dialect Identification, DID)

Working map of the thesis: every chapter, and which repo artifact backs each claim.
Companion drafts: [01-methodology.md](01-methodology.md), [02-results.md](02-results.md),
[03-discussion.md](03-discussion.md).

## 1. Introduction
- Motivation: Bangla dialect diversity; low-resource speech processing; DID as a task
- Problem statement: 9-way classification from 5-second clips
- Contributions (write these last, but they are):
  1. A 13-hour, 9-class Bangla dialect corpus pipeline with leakage-aware splitting
  2. First (or among first) application of self-supervised multilingual speech models
     (Whisper, XLS-R) to Bangla regional DID
  3. **Quantified speaker/channel leakage** via an ECAPA pseudo-speaker strict split —
     turning the usual "upper bound" disclaimer into a measured 12–14 point effect
  4. Ablation across representations × split protocols + data-efficiency analysis
  5. A negative result on channel-suppression augmentation, with a prosody-based
     explanation specific to DID

## 2. Background & Related Work  → draft + paper list in [04-literature-review.md](04-literature-review.md)
- DID literature; accent/language ID; x-vector/ECAPA speaker embeddings
- Self-supervised speech models: wav2vec 2.0 → XLS-R; Whisper (weakly supervised)
- Prior Bangla accent work: the v1 hand-crafted approach and its public replication
  (this motivates the pivot — cite the replication)
- Leakage in audio classification benchmarks (recording-channel shortcuts)

## 3. Dataset
- Source: TV, film, online media; ~5 s clips; 16 kHz mono PCM-16
- 9,305 raw / 9,303 validated clips, ~13.1 h; class table (750–1,654 per class)
- The "Formal" design choice: a register, not a region — argued explicitly
- Metadata reality: 559/1,654 Formal clips have recoverable YouTube source IDs
  (35 sources); regional classes have none → the central leakage problem
- Artifacts: `build_manifest.py`, class table in README

## 4. Methodology  → draft in [01-methodology.md](01-methodology.md)
- Splitting: canonical clip-level split (75/10/15) vs. strict pseudo-speaker split
- ECAPA clustering + threshold calibration on Formal's true sources
- Models: MFCC+MLP (v1 re-implementation), frozen Whisper-small + MLP, XLS-R-300M
  fine-tuning; one shared evaluation protocol (weighted CE, macro-F1, seed 42)
- Compute: M1 laptop + free Kaggle T4 (zero-hardware-budget reproducibility)

## 5. Results  → draft in [02-results.md](02-results.md)
- 5.1 Clip-level baseline (0.917) and fine-tuning (0.952); per-class; confusions
- 5.2 Strict-split results: the 6-cell ablation table; leakage quantified
- 5.3 Data efficiency: 10 % of labels with SSL beats 100 % with MFCC
- 5.4 Augmentation study: negative result (0.808 → 0.748)
- Figures: reports/figures/*.png (confusion matrices, per-class comparison, t-SNE,
  training curve, ablation bars, data-efficiency curves)

## 6. Discussion  → draft in [03-discussion.md](03-discussion.md)
- What the strict split reveals (leakage decomposition; which classes were inflated)
- Errors follow dialect geography (Barishal↔Noakhali; Mymensingh transitional)
- Why augmentation failed: prosody is signal in DID
- Limitations: pseudo-speaker approximation (ARI 0.347), lumpy strict test cells,
  5 s context, single corpus/domain
- Future work: prosody-preserving augmentation, MMS/larger XLS-R, longer context,
  true speaker labels

## 7. Conclusion
- Restate contributions with final numbers; honest headline = strict split

## Appendices
- A: Reproducibility (env, seeds, split files, commands) → [reproducibility.md](../reproducibility.md)
- B: Full per-class tables and confusion matrices (reports/phase2, reports/phase3)
- C: Hyperparameters (lifted from scripts' argparse defaults)
