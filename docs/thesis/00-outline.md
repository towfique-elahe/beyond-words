# Thesis Outline — Beyond Words

*A Neural Network Approach to Classifying Bangla Accentual Diversity*
(framed as Dialect Identification, DID)

Working map of the thesis: every chapter, and which repo artifact backs each claim.
Companion drafts: [01-methodology.md](01-methodology.md), [02-results.md](02-results.md),
[03-discussion.md](03-discussion.md).

**Framing (settled Oct 2026): two routes to dialect identity.** The acoustic
route (Phases 1–3: the *sound* — log-mel frames or the raw waveform, no notion of
what is said) and the lexical route (Phase 5: the *words* — ASR transcripts,
regional word lists) are presented side by side under one question: *is Bangla
dialect identity carried by the words or by the sound?* Fusion of the two is the
combined result. The title "Beyond Words" is read literally.

## 1. Introduction
- Motivation: Bangla dialect diversity; low-resource speech processing; DID as a task
- Problem statement: 9-way classification from 5-second clips
- The two-route question (words vs sound); why both routes are needed to answer it
- Contributions (write these last, but they are):
  1. A 13-hour, 9-class Bangla dialect corpus pipeline with leakage-aware splitting
  2. First (or among first) application of self-supervised multilingual speech models
     (Whisper, XLS-R) to Bangla regional DID
  3. **Quantified speaker/channel leakage** via an ECAPA pseudo-speaker strict split —
     turning the usual "upper bound" disclaimer into a measured 12–14 point effect
  4. Ablation across representations × split protocols + data-efficiency analysis
  5. A negative result on channel-suppression augmentation, with a prosody-based
     explanation specific to DID
  6. **The lexical route**: five off-the-shelf Bangla ASR front-ends compared by how
     much dialect they preserve (automated dialect-word recall on Ben-10, 1,666
     human-transcribed clips), regional-word matching with an interpretable
     per-region match percentage, a TF-IDF text classifier, and acoustic+lexical
     fusion (strict 0.808 → 0.856). Finding: standard-Bangla ASR erases the
     dialect; words alone trail the acoustics but are complementary to them.

## 2. Background & Related Work  → draft + paper list in [04-literature-review.md](04-literature-review.md)
- DID literature; accent/language ID; x-vector/ECAPA speaker embeddings
- Self-supervised speech models: wav2vec 2.0 → XLS-R; Whisper (weakly supervised)
- Prior Bangla accent work: the v1 hand-crafted approach and its public replication
  (this motivates the pivot — cite the replication)
- Leakage in audio classification benchmarks (recording-channel shortcuts)
- Bangla ASR and dialect speech resources: OOD-Speech, the Bengali.AI competition
  models, Ben-10 and its ASR benchmark (the source of the Phase 5 front-ends),
  Vashantor; text-based Bangla dialect classification (BanglaDial) as the
  lexical route's precedent

## 3. Dataset
- Source: TV, film, online media; ~5 s clips; 16 kHz mono PCM-16
- 9,303 usable clips (~13.1 h; 2 duplicate .ogg files excluded); 750–1,652 per class
- The "Formal" design choice: a register, not a region — argued explicitly
- Metadata reality: 557/1,652 Formal clips have recoverable YouTube source IDs
  (35 sources); regional classes have none → the central leakage problem
- Artifacts: `build_manifest.py`, class table in README

## 4. Methodology  → draft in [01-methodology.md](01-methodology.md)
- Splitting: canonical clip-level split (75/10/15) vs. strict pseudo-speaker split
- ECAPA clustering + threshold calibration on Formal's true sources
- Models: MFCC+MLP (v1 re-implementation), frozen Whisper-small + MLP, XLS-R-300M
  fine-tuning; one shared evaluation protocol (weighted CE, macro-F1, seed 42)
- Compute: M1 laptop + free Kaggle T4 (zero-hardware-budget reproducibility)
- The lexical route (§4.7): ASR front-ends (no fine-tuning), standard-Bangla
  vocabulary filter (OOD-Speech), external and in-domain regional lexicons,
  word matching / TF-IDF / late fusion, Ben-10 valid as the ASR reference set

## 5. Results  → draft in [02-results.md](02-results.md)
- 5.1 Clip-level baseline (0.917) and fine-tuning (0.952); per-class; confusions
- 5.2 Strict-split results: the 6-cell ablation table; leakage quantified
- 5.3 Data efficiency: 10 % of labels with SSL beats 100 % with MFCC
- 5.4 Augmentation study: negative result (0.808 → 0.748)
- 5.5 The lexical route: ASR front-ends on Ben-10 valid (WER, dialect-word recall);
  OOV diagnostics on the thesis clips; word matching (0.36) / TF-IDF (0.63) /
  fusion (0.856) under the strict split, with group-bootstrap CIs
- Figures: reports/figures/*.png (confusion matrices, per-class comparison, t-SNE,
  training curve, ablation bars, data-efficiency curves)

## 6. Discussion  → draft in [03-discussion.md](03-discussion.md)
- What the strict split reveals (leakage decomposition; which classes were inflated)
- Errors follow dialect geography (Barishal↔Noakhali; Mymensingh transitional)
- Why augmentation failed: prosody is signal in DID
- Words or sound? The ASR ceiling (≤ 19 % of dialect words survive transcription),
  why the formal-word filter underperforms a plain text classifier, where the
  routes are complementary (Barishal, Sylhet) and where they are not (Khulna)
- Limitations: pseudo-speaker approximation (ARI 0.347), lumpy strict test cells,
  5 s context, single corpus/domain; for the lexical route: no reference
  transcripts on the thesis audio, content leakage not controlled, substitute
  wav2vec2 front-end
- Future work: prosody-preserving augmentation, MMS/larger XLS-R, longer context,
  true speaker labels; dialect-aware ASR as the lexical route's bottleneck

## 7. Conclusion
- Restate contributions with final numbers; honest headline = strict split
  (0.808 acoustic, 0.856 with text); answer the two-route question: on 5 s
  clips the dialect is mostly beyond the words

## Appendices
- A: Reproducibility (env, seeds, split files, commands) → [reproducibility.md](../reproducibility.md)
- B: Full per-class tables and confusion matrices (reports/phase2, reports/phase3)
- C: Hyperparameters (lifted from scripts' argparse defaults)
