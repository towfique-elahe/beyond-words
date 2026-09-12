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
2. **The honest headline for this corpus is 0.808** (fine-tuned XLS-R, strict).
   That remains a strong 9-way result (~9× chance) on genuinely unseen
   pseudo-speakers.
3. Class-level inflation is uneven: Khulna's near-perfect clip-level score
   (0.986) survives the strict split for the baseline (0.887–0.986 range across
   models) less dramatically than Barishal's collapse (0.871 → 0.603), suggesting
   Barishal's clip-level score leaned most on recording-channel signatures.

## 6.2 Errors follow dialect geography

Under every model and protocol, the dominant confusion is Barishal ↔ Noakhali —
two southern coastal dialects with documented linguistic proximity — and
Mymensingh, a transitional zone, disperses errors across its neighbors. This is
linguistically coherent behavior: the models' failure modes recapitulate the
dialect continuum rather than arbitrary decision boundaries, supporting the claim
that they learn accentual structure (and not only channel shortcuts, even on the
clip-level split).

## 6.3 Why fine-tuning's advantage shrinks under strict evaluation

Fine-tuned XLS-R gains +3.5 points at clip level but only +1.5 under the strict
split, and its strict drop (−14.4) slightly exceeds the frozen baseline's (−12.4).
End-to-end adaptation gives the model freedom to exploit *whatever* discriminates
training classes — including channel signatures correlated with class in the
training data. A frozen general-purpose representation cannot specialize this way.
The practical guidance: when leakage cannot be ruled out, frozen-embedding
baselines are not merely cheaper — they are partially self-regularizing against
shortcut learning, and the fine-tuning margin measured on a leaky split will
overstate the deployable improvement.

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

## 6.5 Limitations

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
- The Formal class is one register from 35 sources; its stability across
  protocols partly reflects that narrowness.

## 6.6 Future work

- Prosody-preserving augmentation chain on the strict split (the direct follow-up)
- True speaker annotation for a subset, to validate the pseudo-speaker bound
- Larger multilingual encoders (XLS-R-1B, MMS) under the same fp16/T4 budget
- Longer-context inputs; clip aggregation to speaker- or recording-level decisions
- Cross-domain evaluation (telephone, conversational speech)
- A public inference demo over the strict-split checkpoint (Phase 4 artifact)
