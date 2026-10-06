"""
Phase 5 - Score the ASR front-ends against human transcripts (Ben-10 valid).

The thesis audio has no reference text, so ASR quality is measured on the
public Ben-10 validation set (1,666 clips, 10 districts, dialect transcripts
written as spoken). Hypotheses come from transcribe.py run on that audio with
a manifest whose relpath basenames are Ben-10's file_name values.

Per front-end, overall and per district:
    wer / cer        corpus-level, on normalised tokens (punctuation ignored)
    dialect_recall   share of reference words OUTSIDE the formal vocabulary that
                     the hypothesis reproduces exactly - the automated version of
                     the Ben-10 paper's hand-counted "dialect recall"
    formal_recall    the same for reference words inside the formal vocabulary

Usage:
    python ben10_eval.py --reference data/ben10/valid.csv \
        transcripts/ben10_valid/tugstugi_ben10.csv transcripts/ben10_valid/tugstugi.csv
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import jiwer
import pandas as pd

from text_utils import load_vocab, tokenize


def recall(ref_tokens, hyp_tokens, keep):
    """Count-matched recall of the reference tokens selected by keep(word)."""
    hyp = Counter(hyp_tokens)
    hit = total = 0
    for w, c in Counter(w for w in ref_tokens if keep(w)).items():
        total += c
        hit += min(c, hyp[w])
    return hit, total


def evaluate(df, formal):
    rows = []
    for district, g in [("all", df)] + list(df.groupby("district")):
        refs = [" ".join(t) for t in g["ref_tok"]]
        hyps = [" ".join(t) for t in g["hyp_tok"]]
        keep = [i for i, r in enumerate(refs) if r]          # jiwer needs non-empty refs
        d_hit = d_tot = f_hit = f_tot = 0
        for r, h in zip(g["ref_tok"], g["hyp_tok"]):
            a, b = recall(r, h, lambda w: w not in formal)
            d_hit, d_tot = d_hit + a, d_tot + b
            a, b = recall(r, h, lambda w: w in formal)
            f_hit, f_tot = f_hit + a, f_tot + b
        rows.append({
            "district": district, "clips": len(g),
            "wer": round(jiwer.wer([refs[i] for i in keep], [hyps[i] for i in keep]), 4),
            "cer": round(jiwer.cer([refs[i] for i in keep], [hyps[i] for i in keep]), 4),
            "dialect_words": d_tot, "dialect_recall": round(d_hit / max(d_tot, 1), 4),
            "formal_recall": round(f_hit / max(f_tot, 1), 4),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hyps", nargs="+", help="transcribe.py CSV(s) on Ben-10 valid audio")
    ap.add_argument("--reference", default="data/ben10/valid.csv")
    ap.add_argument("--formal", default="lexicons/formal_vocab.txt")
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--out", default="reports/phase5/ben10_valid_eval.json")
    args = ap.parse_args()

    formal = load_vocab(args.formal, args.min_count)
    ref = pd.read_csv(args.reference)
    ref["ref_tok"] = ref["transcripts"].map(tokenize)
    report = {"reference": args.reference, "formal": args.formal, "min_count": args.min_count,
              "fronts": {}}
    for path in args.hyps:
        hyp = pd.read_csv(path)
        hyp["file_name"] = hyp["relpath"].map(os.path.basename)
        df = ref.merge(hyp[["file_name", "text"]], on="file_name", how="left")
        missing = int(df["text"].isna().sum() - hyp["text"].isna().sum())
        df["hyp_tok"] = df["text"].map(tokenize)
        table = evaluate(df, formal)
        name = Path(path).stem
        print(f"\n=== {name}  ({missing} reference clips without a hypothesis) ===")
        print(table.to_string(index=False))
        report["fronts"][name] = {"missing": missing, "table": table.to_dict("records")}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
