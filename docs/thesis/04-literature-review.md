# Chapter Draft — Literature Review (Related Work)

Structure for Chapter 2, with an annotated paper list per section and the role
each source plays in this thesis's argument. Verify every citation's final venue
and author list before submission (arXiv versions can differ from published ones);
entries marked ⚑ were located by web search in Sep 2026 — double-check details,
the rest are canonical and safely citable.

## Suggested narrative arc

1. DID sits between language ID and speaker ID (§2.1) →
2. hand-crafted features gave way to self-supervised representations (§2.2) →
3. Bangla specifically is low-resource and understudied for DID (§2.3) →
4. found-data corpora invite channel/speaker shortcuts, and the field under-reports
   this (§2.4) →
5. therefore: SSL models + leakage-aware evaluation on Bangla = the gap this
   thesis fills.

---

## 2.1 Dialect and accent identification

- **Snyder et al., 2018 — "X-vectors: Robust DNN embeddings for speaker
  recognition" (ICASSP).** The embedding paradigm that modern DID/LID systems
  build on; frames dialect ID's kinship with speaker ID — which is exactly why
  speaker leakage inflates DID scores (bridge to §2.4).
- **Ali et al., 2019 — MGB-5 / ADI17 challenge (ASRU): fine-grained Arabic
  dialect identification from audio.** The reference large-scale DID task; the
  17-way Arabic setting is the closest analogue to 9-way Bangla.
- ⚑ **Khaled & Elnagar, 2025 — "Arabic Dialect Audio Identification Using
  Wav2Vec2" (Springer).** Fine-tuned XLSR on MGB-3; ~93 % on region/country —
  directly comparable design to this thesis's Phase 2.
  https://link.springer.com/chapter/10.1007/978-3-031-80438-0_6
- ⚑ **"On the Robustness of Arabic Speech Dialect Identification" (arXiv
  2306.03789).** Robustness analysis of DID systems — supports the claim that
  DID scores degrade sharply off the training channel/domain.
  https://arxiv.org/pdf/2306.03789
- ⚑ **Hämäläinen et al., "Finnish Dialect Identification: The Effect of Audio
  and Text" (arXiv 2111.03800).** DID for another lower-resource language;
  useful for positioning audio-only DID difficulty.
  https://arxiv.org/pdf/2111.03800
- **Valk & Alumäe, 2021 — VoxLingua107 (SLT).** Large-scale spoken LID with
  found YouTube data; relevant both as method and as an example of web-scraped
  corpora (channel confounds included).
- ⚑ **"Improving Language Identification of Accented Speech" (arXiv
  2203.16972).** The LID↔accent interaction; motivates treating register
  (Formal) as a class.  https://arxiv.org/pdf/2203.16972

## 2.2 Self-supervised and weakly supervised speech representations

- **Baevski et al., 2020 — wav2vec 2.0 (NeurIPS).** The SSL architecture behind
  XLS-R; cite for the pretraining objective and fine-tuning recipe.
- **Babu et al., 2022 — XLS-R: Self-supervised cross-lingual speech
  representation learning at scale (Interspeech; arXiv 2111.09296).** The exact
  model fine-tuned in Phase 2 (300M variant); pretrained on 128 languages
  including Bangla — the multilingual-coverage argument for choosing it.
- **Radford et al., 2023 — Whisper (ICML; arXiv 2212.04356).** Weakly supervised
  encoder used frozen in Phase 1; cite for the encoder architecture and its
  multilingual training (Bangla included).
- **Yang et al., 2021 — SUPERB benchmark (Interspeech).** Evidence that frozen
  SSL representations transfer across many speech tasks with light heads — the
  design rationale for the Phase 1 cached-embedding baseline.
- ⚑ **Yang et al., 2023 — "What Can an Accent Identifier Learn? Probing Phonetic
  and Prosodic Information in a Wav2vec2-based Accent Identification Model"
  (arXiv 2306.06524).** Shows accent-ID fine-tuning reshapes *prosodic* encoding
  in wav2vec2 layers — **key external support for this thesis's explanation of
  the augmentation negative result** (pitch/tempo perturbations destroy
  dialect-bearing prosody).  https://arxiv.org/pdf/2306.06524
- **Desplanques et al., 2020 — ECAPA-TDNN (Interspeech).** The speaker-embedding
  model used for pseudo-speaker clustering in Phase 3.

## 2.3 Bangla speech and dialect resources

- ⚑ **"Bengali Accent Classification from Speech Using Different Machine
  Learning and Deep Learning Techniques" (Springer, 2020/21).** The closest
  prior line to v1 (hand-crafted features + classical/DL classifiers); position
  the thesis against it and note the public replication of the v1-style
  approach that motivated the v2 pivot.
  https://link.springer.com/chapter/10.1007/978-981-15-7394-1_46
- ⚑ **"Bangla Dialect Classification Across 11 Regions Using Machine Learning
  and Transformer-Based Models" (IEEE, 2025).** Text-based dialect
  classification (BanglaDial; BanglaBERT ~88.7 %) on *human-written* dialect
  text — the precedent for the lexical route, and the contrast case: Phase 5
  shows what happens when the text has to come from ASR instead (§2.6).
  https://ieeexplore.ieee.org/document/11429342/
- ⚑ **Bangla Regional Dialects Speech Dataset (Mendeley Data).** 5-region
  speech dataset; compare scale/coverage with this thesis's 9-class ~13 h corpus.
  https://data.mendeley.com/datasets/777wsgjgtm/1
- ⚑ **BRADS & BRWDS: audio+text datasets for Bangla regional speech recognition
  (Data in Brief / PMC).** Recent multipurpose regional resources.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12594939/
- ⚑ **Alam et al. — Bengali Common Voice Speech Dataset (arXiv 2206.14053).**
  ~400 h crowdsourced Bangla ASR corpus with accent metadata; the main
  large-scale Bangla speech resource to acknowledge.
  https://arxiv.org/pdf/2206.14053
- ⚑ **"Challenges and Opportunities of Speech Recognition for Bengali"
  (arXiv 2109.13217).** Survey framing Bangla as low-resource with high
  dialectal diversity — the motivation citation for the introduction.
  https://arxiv.org/pdf/2109.13217

## 2.4 Leakage, shortcuts, and honest evaluation

- **Geirhos et al., 2020 — "Shortcut learning in deep neural networks" (Nature
  Machine Intelligence).** The general framework: models exploit whatever
  discriminates training classes; cite when interpreting the fine-tuning drop
  under the strict split.
- ⚑ **SpurAudio: benchmark for shortcut learning in few-shot audio
  classification (arXiv 2605.13672).** Recent, audio-specific shortcut-learning
  evidence.  https://arxiv.org/pdf/2605.13672
- ⚑ **EchoHawk (arXiv 2606.29589) — cautionary study of session-level data
  leakage in acoustic drone detection.** Clips sharing a session share channel
  and background; session-disjoint evaluation collapses inflated scores —
  the same phenomenon this thesis measures for DID.
  https://arxiv.org/pdf/2606.29589
- ⚑ **"Benchmarking Data Leakage and Generalization in Audio Classification"
  (ICIIS 2025).** Field-level evidence that evaluation protocols are
  under-specified in audio-classification papers.
  https://asayakkara.org/downloads/publications/Benchmarking-Data-Leakage-ICIIS-2025.pdf

## 2.5 Data augmentation for speech

- **Ko et al., 2015 — "Audio augmentation for speech recognition"
  (Interspeech).** Speed perturbation as a standard, *beneficial* ASR
  augmentation — the foil for this thesis's negative result: what helps ASR
  (prosody-invariant targets) hurts DID (prosody-bearing targets).
- **Park et al., 2019 — SpecAugment (Interspeech).** The canonical
  augmentation-for-speech citation; note it operates on features, whereas this
  thesis's chain operates on waveforms.

## 2.6 Bangla ASR and dialect speech resources (the lexical route)

- ⚑ **Dipto et al., 2025 — "Are ASR foundation models generalized enough to
  capture features of regional dialects for low-resource languages?"
  (IJCNLP-AACL 2025 short, pp. 178–188; arXiv 2510.23252).** Introduces
  **Ben-10** (78 h, 16,690 clips, 394 speakers, 10 districts, dialect
  transcribed as spoken) and benchmarks seven ASR systems on it by WER/CER:
  Whisper-large-v3 1.13, Google ASR 1.03, Hishab Conformer 0.87, Wav2Vec2
  (SCB) 0.89, Tugstugi 0.81, and the two Ben-10-fine-tuned variants 0.83 /
  **0.70**. Its §5 hand-counts "dialect recall" on ~50 words per district
  (0.01–0.09 zero-shot vs 0.16–0.53 after Ben-10 fine-tuning) and calls the
  full-scale version "beyond the scope of this paper" — Phase 5's automated
  dialect-word recall on the full validation set is that version. **It is an
  ASR benchmark, not a dialect-identification paper**: no classifier, no
  accuracy or F1 — its models are this thesis's transcription front-ends, not a
  DID baseline. Code: github.com/BengaliAI/reg-speech-aacl (includes per-district
  TF-IDF word lists). Dataset: HF `bengaliAI/Ben-10` (CC0; test set private).
  https://aclanthology.org/2025.ijcnlp-short.17/
- ⚑ **Rakib et al., 2023 — "OOD-Speech: A Large Bengali Speech Recognition
  Dataset for Out-of-Distribution Benchmarking" (Interspeech 2023; arXiv
  2305.09688).** 1,178 h of crowd-read standard Bangla from 22,645 speakers,
  plus a 23 h test set from 17 out-of-distribution domains. Its training
  transcripts (Kaggle competition `bengaliai-speech`) are Phase 5's
  standard-Bangla vocabulary; the Ben-10 paper defines district OOV rates
  against the same corpus.  https://arxiv.org/abs/2305.09688
- ⚑ **Bengali.AI Speech Recognition competition, 2023 (Kaggle).** The
  first-place "Tugstugi" solution — a Whisper-medium fine-tune on OOD-Speech,
  released as `bengaliAI/tugstugi_bengaliai-asr_whisper-medium` — and its
  Ben-10 fine-tune `…-regional-asr_whisper-medium` are two of the five
  front-ends. Cite the competition and the model cards.
- ⚑ **Nandi et al., 2023 — Hishab Bangla FastConformer** (`hishab/hishab_bn_fastconformer`,
  NeMo, CC-BY-NC-4.0; ~18k h MegaBNSpeech, mostly news). The industrial
  standard-Bangla front-end; in Phase 5 it has the lowest out-of-vocabulary
  rate of all five on regional clips — i.e. it standardises the dialect most.
- ⚑ **Radford et al., 2023 — Whisper (ICML).** Already cited for the Phase 1
  encoder; in Phase 5 `openai/whisper-large-v3` is the zero-shot front-end and
  the worst on dialect speech (WER 0.96 on Ben-10 valid, repetition loops in
  12 % of thesis clips) — the generalisation question in Dipto et al.'s title.
- ⚑ **Faria et al., 2023 — "Vashantor: A Large-scale Multilingual Benchmark
  Dataset for Automated Translation of Bangla Regional Dialects to Bangla
  Language" (arXiv 2311.11142).** 32,500 sentence pairs, dialect ↔ standard
  Bangla (plus Banglish and English) for Chittagong, Noakhali, Sylhet,
  Barishal and Mymensingh; CC BY 4.0 (Mendeley `bj5jgk878b`). The parallel
  structure lets a regional word be defined as "present on the dialect side,
  absent on the standard side" — the external lexicon's second source.
  https://arxiv.org/abs/2311.11142
- **RegSpeech12 (arXiv 2510.24096)** — same group as Ben-10, 12 dialects
  (adds Comilla and Noakhali), ~100 h; and **BanglaDialecto (arXiv 2411.10879)**
  — Noakhali speech → dialect text → standard translation. Mention as the
  resources that would extend external lexicons to more of this corpus's
  classes; not used.
- **Software:** `bnunicodenormalizer` (Bangla Unicode normalisation), `jiwer`
  (WER/CER), NVIDIA NeMo, Hugging Face Transformers.

---

## How to use this list

- §2.3 papers set up the gap: Bangla dialect work is mostly text-based or
  hand-crafted-feature audio work on small corpora; none combines SSL models
  with leakage-aware evaluation.
- §2.4 papers legitimize the strict split as the chapter's methodological stake,
  and EchoHawk gives precedent for reporting the clip-level → group-disjoint gap
  as a finding in itself.
- The probing paper (2306.06524) plus Ko et al. (2015) together carry the
  discussion of the augmentation negative result.
- §2.6 sets up the lexical route: BanglaDial shows text-based DID works on
  human-written dialect text; Dipto et al. show ASR does not produce such text
  for standard-trained models; Phase 5 connects the two. Be explicit that the
  Ben-10 paper's WER numbers are not comparable to DID accuracy and that its
  models are used as front-ends, not as competing DID systems.
- Remember to also cite software: PyTorch, Hugging Face Transformers,
  SpeechBrain, librosa, scikit-learn, audiomentations.
