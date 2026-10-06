"""
Phase 5 - How much dialect signal does each ASR front-end leave in its output?

The dataset has no reference transcripts, so WER cannot be measured on it.
What can be measured without references, per front-end and per district:
    empty      share of clips with no Bangla word in the transcript
    tok/clip   mean word tokens per clip
    oov        share of tokens outside the formal (standard-Bangla) vocabulary
    clips>=1   share of clips with at least one out-of-vocabulary token
A front-end that rewrites dialect speech into standard Bangla shows a low oov
on the regional classes, close to the Formal class.

Works on any CSV with a text column and a district column, so the same table
can be produced for human transcripts (e.g. Ben-10 valid.csv) as a reference
point for what real dialect text looks like against the same vocabulary.

Usage:
    python asr_diagnostics.py transcripts/tugstugi_ben10.csv transcripts/tugstugi.csv \
        --formal lexicons/formal_vocab.txt --out reports/phase5/pilot_diagnostics.json
    python asr_diagnostics.py data/ben10/valid.csv --text-col transcripts \
        --formal lexicons/formal_vocab.txt
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from text_utils import load_vocab, tokenize


def diagnose(df, text_col, formal, top):
    """Per-district table plus the most frequent out-of-vocabulary words."""
    rows, top_oov = [], {}
    for district, g in df.groupby("district"):
        toks = [tokenize(t) for t in g[text_col]]
        oov = [[w for w in ts if w not in formal] for ts in toks]
        n_tok = sum(map(len, toks))
        rows.append({
            "district": district,
            "clips": len(g),
            "empty": sum(not ts for ts in toks) / len(g),
            "tok_per_clip": n_tok / len(g),
            "oov": sum(map(len, oov)) / max(n_tok, 1),
            "clips_with_oov": sum(bool(o) for o in oov) / len(g),
        })
        counts = Counter(w for o in oov for w in o)
        top_oov[district] = [w for w, _ in counts.most_common(top)]
    return pd.DataFrame(rows), top_oov


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", help="CSV(s) with a district and a text column")
    ap.add_argument("--formal", default="lexicons/formal_vocab.txt")
    ap.add_argument("--min-count", type=int, default=1, help="formal-vocab count threshold")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--top", type=int, default=15, help="OOV words to list per district")
    ap.add_argument("--out", default="", help="optional JSON path")
    args = ap.parse_args()

    formal = load_vocab(args.formal, args.min_count)
    print(f"formal vocabulary: {len(formal):,} words ({args.formal}, count >= {args.min_count})")

    report = {}
    for path in args.transcripts:
        table, top_oov = diagnose(pd.read_csv(path), args.text_col, formal, args.top)
        name = Path(path).stem
        print(f"\n=== {name} ===")
        print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        for district, words in top_oov.items():
            print(f"  {district:<11} {' '.join(words)}")
        report[name] = {"table": table.to_dict("records"), "top_oov": top_oov}

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"formal": args.formal, "min_count": args.min_count,
                       "fronts": report}, f, ensure_ascii=False, indent=2)
        print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
