# Chapter Draft — Discussion

## 6.1 What the strict split reveals

The central methodological contribution is converting a disclaimer into a
measurement. Most audio-classification work on found data (TV, YouTube) either
ignores speaker/channel leakage or discloses it qualitatively. Here, the
pseudo-speaker strict split bounds it: 12–14 macro-F1 points for pretrained
representations, ~25 for hand-crafted features. Three implications:

1. **Reported state-of-the-art numbers on clip-level splits of found-data corpora
   are inflated**, and the inflation is representation-dependent — comparisons
   between papers using different features on clip-level splits partially rank
   channel exploitation, not dialect skill.
2. **The honest headline for this corpus is 0.92** (fine-tuned Whisper-medium /
   Tugstugi Ben-10, strict; Phase 6), ~8× chance on genuinely unseen
   pseudo-speakers; XLS-R, on which the leakage analysis was built, reaches 0.808.
3. Class-level inflation is uneven, but per-class strict numbers must be read
   with care: Barishal's strict test set is a single 251-clip pseudo-speaker
   cluster, so its collapse (0.871 → 0.603 baseline) measures generalization to
   one speaker/source rather than to the class. Macro-level conclusions are the
   robust ones.

## 6.2 Errors follow dialect geography

Under every model and protocol, the dominant confusion is Barishal ↔ Noakhali —
two southern coastal dialects with documented linguistic proximity — and
Mymensingh, a transitional zone, disperses errors across its neighbors. This is
linguistically coherent behavior: the models' failure modes recapitulate the
dialect continuum rather than arbitrary decision boundaries, supporting the claim
that they learn accentual structure (and not only channel shortcuts, even on the
clip-level split).

## 6.3 Frozen or fine-tuned, and which encoder? (revised after Phase 6)

The Phase 3 comparison (frozen Whisper-small vs fine-tuned XLS-R: +3.5 clip,
+1.5 strict n.s., larger drop for the fine-tuned model) suggested fine-tuning
mainly buys channel exploitation. Phase 6 shows that was an encoder effect:
- On one checkpoint, fine-tuning helps clearly (Whisper-medium: 0.782 → 0.919
  strict, CI +6.5 to +22.3; Tugstugi Ben-10: 0.857 → 0.921, CI +3.6 to +10.4)
  and *reduces* the leakage drop (13.8 → 6.5; 7.9 → 6.0).
- XLS-R (contrastive SSL on raw audio) is the channel exploiter (drop 14.4);
  the transcription-pretrained Whisper-medium family drops 6–7 and reaches
  0.92 on unseen speakers.
- Bangla-specific pretraining helps frozen (Tugstugi Ben-10 0.857 vs
  Whisper-medium 0.782, significant) but adds nothing once fine-tuned
  (Whisper-medium ≈ Tugstugi ≈ Tugstugi Ben-10); architecture/size matter more (Hishab
  FastConformer, 115M, CTC, news: 0.746 strict, drop 15.5; possibly
  under-trained, best epoch 18/20).
Guidance: pick an encoder pretrained to transcribe, then fine-tune; a frozen
baseline still guards against shortcuts but is not a substitute. The honest
headline becomes 0.92 (§6.1 point 2 updated accordingly).

## 6.4 The augmentation negative result: prosody is signal in DID

The channel-suppression chain reduced strict macro-F1 by 6 points (0.808 → 0.748).
The chain included pitch shift (±2 st) and time stretch (0.9–1.1) — standard,
beneficial transforms in ASR, where the lexical target is invariant to prosody.
DID inverts this: pitch contour, rhythm, and tempo are *dialect-bearing cues*, so
these transforms corrupt labels' acoustic support rather than removing nuisance
variation. Depressed validation throughout training (0.702 vs. 0.757 best val)
indicates a genuinely harder-to-fit target, not an unlucky checkpoint.

This yields a task-specific recommendation: augmentation policies for dialect and
accent identification should be **prosody-preserving** (additive noise, EQ,
band-limiting, gain), and prosodic perturbations should be treated as harmful
until shown otherwise. Testing that reduced chain is left as future work.

## 6.5 Words or sound? What the lexical route shows

The thesis title is read literally in Phase 5: how much of Bangla dialect
identity is *in the words*, and how much lies beyond them? Three findings.

1. **Standard-Bangla ASR removes the dialect before a classifier can see it.**
   Models trained on standard colloquial Bangla (Tugstugi, Hishab) transcribe
   regional speech into standard forms — "বিয়া দিয়া দাও" becomes "বিয়ে দিয়ে
   দাও" — and reproduce only 1–4 % of the dialect words in Ben-10's human
   transcripts. Their output on regional clips is barely less "formal" than on
   Formal clips. The one front-end tuned on dialect speech reproduces 27 %,
   which is both the best available and a hard ceiling for any word-list
   method: nearly three of four dialect words never reach the text. This is the
   Ben-10 paper's hand-counted observation, confirmed on the full validation
   set with an automated measure, and it is the single most important fact
   about the lexical route.

2. **Filtering to regional words discards evidence.** The intuitive design —
   drop every word that standard Bangla also has, match what is left against
   regional word lists — scores 0.36 under the strict split, because fewer
   than half of the 5-second clips retain any matchable word. A plain TF-IDF
   classifier over *all* words and character n-grams reaches 0.63 with the
   same transcripts. The difference is what the filter throws away: dialect
   verb morphology (-তাছি, -মু, -ইন, -ছইন), pronoun forms, and function words
   that are in the standard vocabulary yet distributed unevenly across
   regions. Word matching keeps its value as an *explanation* (the demo shows
   which words decided a clip and how strongly each belongs to each region),
   not as the classifier.

3. **Words and sound are complementary, modestly.** Fusion lifts the honest
   headline from 0.808 to 0.856 (+4.8 strict; borderline by the speaker-group
   bootstrap, with a 4 % probability of no gain) and 0.952 to 0.967 at clip
   level (significant). The gain lands where the acoustic model is weakest
   under the strict split — Barishal (+21 points, the single-cluster test
   cell) and Sylhet (+8, the dialect with the most distinctive and
   best-transcribed vocabulary) — and is negative only for Khulna, where the
   acoustics are already near-perfect. The two routes fail on different clips,
   which is the condition under which fusion helps; that they do so is itself
   evidence that the text carries dialect information and not only a noisier
   copy of the acoustic decision.

The answer to the question, for clips this short: **mostly beyond the words.**
Words alone (0.63) sit between hand-crafted acoustics (0.48) and frozen Whisper
(0.79); the acoustic route is the stronger single route, and the lexical route's
ceiling is set by ASR, not by the classifier. Two caveats temper the comparison.
The TF-IDF model may partly learn programme- or topic-specific vocabulary — the
strict split separates speakers, not content — so 0.63 is itself an upper bound
on *dialect* information in the words. And the acoustic models were trained on
13 h of in-domain audio, whereas the ASR front-ends saw none; a dialect-aware ASR
trained on this corpus would raise the lexical ceiling, at the cost of the
"off-the-shelf, no fine-tuning" setting that makes the comparison clean.

## 6.6 Limitations

- **Pseudo-speakers are approximate.** Calibration against Formal's true sources
  reaches ARI 0.347; clustering both over-merges (a 251-clip Barishal cluster)
  and over-splits (many singletons). The strict split removes much, not provably
  all, speaker/channel overlap — true speaker labels would tighten the bound in
  both directions.
- **Lumpy strict test cells.** Group-disjoint splitting with imbalanced clusters
  concentrates some classes' test sets in few clusters; per-class strict numbers
  carry high variance (macro-average conclusions are robust; single-class strict
  comparisons are not).
- **5-second clips** exclude longer-context dialect cues (discourse markers,
  intonation phrases).
- **Single-domain corpus** (broadcast/online media); generalization to
  conversational or telephone speech is untested.
- **Phase 6 models are single runs** (one seed, one schedule); Hishab may be
  under-trained.
- **Fusion helps only a weak acoustic model**: with Whisper-medium the words
  lower strict macro-F1 0.919 → 0.909 (CI −2.5 to −0.03); the transcript
  information is largely a subset of what the strong encoder extracts. Only
  late log-linear fusion was tried.
- **Single-seed fine-tuning**: XLS-R results are one seed each; bootstrap CIs
  cover test-set sampling noise, not training-seed variance.
- **Whisper pooling includes padding**: Whisper's encoder always sees 30 s of
  (padded) input, and the baseline mean-pools all 1,500 frames, of which only
  ~250 cover the 5 s of speech. The padding contribution is constant across
  clips and the features are standardized, but pooling only the speech frames is
  the cleaner design.
- **Found-data authenticity**: clips from films and dramas may include actors
  performing a dialect rather than native speakers.
- **Lexical route — no reference transcripts on the thesis audio.** ASR
  quality is measured on Ben-10 valid, a different corpus (longer clips,
  spontaneous speech, only three districts in common); WER on the thesis
  clips is unknown. Hand-transcribing a gold subset (~20 clips per class) would
  close this gap and was deferred.
- **Lexical route — content leakage is not controlled.** The strict split is
  speaker-disjoint, not topic-disjoint; recurring programmes, character names
  and topics can appear on both sides and inflate the text classifier. The
  in-domain lexicon's ≥ 2-pseudo-speaker rule and the external-lexicon variant
  mitigate but do not remove this.
- **Lexical route — front-end substitutions.** The paper's "Wav2Vec2 (SCB)"
  checkpoint is not public; `arijitx/wav2vec2-xls-r-300m-bengali` stands in and
  is not the same model. Google ASR and the paper's Ben-10-tuned wav2vec2 were
  not run. The Ben-10-tuned Tugstugi model was trained on ten districts, of
  which only Barishal, Chattogram and Sylhet overlap this corpus; Khulna,
  Rajshahi and Dhaka have no clean external word list at all.
- **Validation-set reuse.** XLS-R's checkpoint was selected on the same
  validation split later used to tune the fusion weight α, so XLS-R's
  validation probabilities are mildly optimistic and α may lean towards the
  acoustic model. Test-set numbers are unaffected; all tuning used val only.
- **Single-seed, greedy decoding.** Transcripts are one greedy decode per
  front-end; beam search or temperature fallback might reduce Whisper-large-v3's
  repetition loops. The committed numbers use the Kaggle T4 transcripts, which
  agree with the M1 pilot on 93.6 % of clips.

## 6.7 Future work

- Prosody-preserving augmentation chain on the strict split (the direct follow-up)
- True speaker annotation for a subset, to validate the pseudo-speaker bound
- Larger multilingual encoders (XLS-R-1B, MMS) under the same fp16/T4 budget
- Longer-context inputs; clip aggregation to speaker- or recording-level decisions
- Cross-domain evaluation (telephone, conversational speech)
- A public inference demo over the strict-split checkpoint (Phase 4 artifact;
  `predict.py --explain` now pairs the acoustic decision with the words behind
  the lexical one)
- Lexical route: a dialect-aware ASR front-end fine-tuned on this corpus (the
  bottleneck is transcription, not classification); a gold-transcribed subset
  for WER on the thesis audio; a topic-disjoint split to bound content leakage;
  joint rather than late fusion (text features into the XLS-R head)
