"""
Phase 5 - Build the word lists used by the lexical (word-level) route.

formal    Standard-Bangla vocabulary from OOD-Speech train.csv (Kaggle
          competition bengaliai-speech; accept the rules on the account first).
          A transcript word outside this vocabulary is a candidate regional word.

external  Regional lexicon from human-written dialect text, independent of the
          thesis audio: Ben-10 train transcripts and the Vashantor parallel
          corpus. Clean, but covers only Barishal, Chattogram, Sylhet, Noakhali
          and Mymensingh (districts are mapped by exact name only).

indomain  Regional lexicon from the ASR transcripts of the TRAIN split only.
          Covers all classes, but is noisy: a word is kept only if it occurs
          at least --min-count times across at least --min-speakers
          pseudo-speakers, which drops ASR garbage and programme-specific words.

A regional lexicon is {region: {word: weight}}. The weight is p(region | word):
the word's size-normalised rate in that region divided by its summed rate over
all regions in the source, so a word shared by several dialects counts less.

Usage:
    python build_lexicon.py formal --source data/ood_speech/train.csv
    python build_lexicon.py external --ben10 data/ben10/train.csv \
        --vashantor data/vashantor/Vashantor_CSV_Format
    python build_lexicon.py indomain --transcripts transcripts/tugstugi_ben10.csv \
        --manifest splits/manifest_split_strict_rel.csv \
        --out lexicons/indomain_tugstugi_ben10_strict.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from build_manifest import extract_source_id
from text_utils import load_vocab, tokenize

# external district name -> thesis class (exact-name matches only)
BEN10_MAP = {"barishal": "Barishal", "chittagong": "Chattogram", "sylhet": "Sylhet"}
VASHANTOR_MAP = {"Barishal": "Barishal", "Chittagong": "Chattogram", "Sylhet": "Sylhet",
                 "Noakhali": "Noakhali", "Mymensingh": "Mymensingh"}
ALPHA = 0.5     # add-alpha smoothing of per-region word rates


# ----------------------------- formal -----------------------------
def load_sentences(source, column):
    """A named column of a CSV, or a 0-based column index of a headerless TSV/CSV."""
    sep = "\t" if source.endswith(".tsv") else ","
    if column.isdigit():
        return pd.read_csv(source, sep=sep, header=None, usecols=[int(column)],
                           quoting=3 if sep == "\t" else 0)[int(column)]
    return pd.read_csv(source, sep=sep, usecols=[column])[column]


def build_formal(args):
    counts = Counter()
    for s in tqdm(load_sentences(args.source, args.column), desc="tokenising"):
        counts.update(tokenize(s))
    kept = [(w, c) for w, c in counts.most_common() if c >= args.min_count]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for w, c in kept:
            f.write(f"{w}\t{c}\n")
    print(f"{sum(counts.values()):,} tokens, {len(counts):,} word types; "
          f"kept {len(kept):,} with count >= {args.min_count} -> {out}")


# ----------------------------- regional -----------------------------
def clip_groups(df):
    """Pseudo-speaker per clip; sourced Formal clips fall back to their YouTube ID."""
    src = df["relpath"].map(extract_source_id)
    spk = df["speaker_id"] if "speaker_id" in df else pd.Series("", index=df.index)
    return spk.fillna("").where(spk.notna() & (spk != ""), src).where(
        lambda g: g != "", df["relpath"])


def to_lexicon(counts, totals, keep):
    """counts: {region: Counter of candidate words}; totals: {region: all tokens};
    keep(region, word) -> bool. Returns {region: {word: p(region | word)}}."""
    regions = sorted(counts)
    lexicon = {}
    for r in regions:
        words = {}
        for w, c in counts[r].items():
            if not keep(r, w):
                continue
            rates = {q: (counts[q][w] + ALPHA) / totals[q] for q in regions}
            words[w] = round(rates[r] / sum(rates.values()), 4)
        lexicon[r] = dict(sorted(words.items(), key=lambda kv: -kv[1]))
    return lexicon


def save_lexicon(lexicon, meta, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "lexicon": lexicon}, f, ensure_ascii=False, indent=1)
    for r, words in lexicon.items():
        print(f"  {r:<11} {len(words):>6} words   {' '.join(list(words)[:10])}")
    print(f"saved -> {out}")


def build_external(args):
    formal = load_vocab(args.formal, args.formal_min_count)
    counts, totals = defaultdict(Counter), Counter()

    if args.ben10:
        df = pd.read_csv(args.ben10)
        df = df[df["district"].isin(BEN10_MAP)]
        for district, text in zip(df["district"], df["transcripts"]):
            toks = tokenize(text)
            totals[BEN10_MAP[district]] += len(toks)
            counts[BEN10_MAP[district]].update(w for w in toks if w not in formal)

    if args.vashantor:
        for path in sorted(Path(args.vashantor).rglob("*.csv")):
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            region = VASHANTOR_MAP[path.name.split()[0]]
            regional_col = next(c for c in df.columns
                                if c.endswith("_bangla_speech") and c != "bangla_speech")
            for standard, regional in zip(df["bangla_speech"], df[regional_col]):
                # a word the standard side of the same pair also uses is not regional
                own_standard = set(tokenize(standard))
                toks = tokenize(regional)
                totals[region] += len(toks)
                counts[region].update(w for w in toks
                                      if w not in formal and w not in own_standard)

    lexicon = to_lexicon(counts, totals, lambda r, w: counts[r][w] >= args.min_count)
    save_lexicon(lexicon, {"kind": "external", "ben10": args.ben10, "vashantor": args.vashantor,
                           "formal": args.formal, "formal_min_count": args.formal_min_count,
                           "min_count": args.min_count}, args.out)


def build_indomain(args):
    formal = load_vocab(args.formal, args.formal_min_count)
    man = pd.read_csv(args.manifest)
    man["group"] = clip_groups(man)
    df = man.merge(pd.read_csv(args.transcripts)[["relpath", "text"]], on="relpath")
    train = df[df["split"] == "train"]
    held_out = set(man.loc[man["split"] != "train", "relpath"])
    assert not held_out & set(train["relpath"]), "val/test clip leaked into the lexicon"
    print(f"{len(train)} train transcripts of {len(df)} matched "
          f"({(man['split'] == 'train').sum()} train clips in the manifest)")

    counts, totals, speakers = defaultdict(Counter), Counter(), defaultdict(set)
    for district, group, text in zip(train["district"], train["group"], train["text"]):
        toks = tokenize(text)
        totals[district] += len(toks)
        for w in toks:
            if w not in formal:
                counts[district][w] += 1
                speakers[district, w].add(group)

    lexicon = to_lexicon(counts, totals, lambda r, w: counts[r][w] >= args.min_count
                         and len(speakers[r, w]) >= args.min_speakers)
    save_lexicon(lexicon, {"kind": "indomain", "transcripts": args.transcripts,
                           "manifest": args.manifest, "train_clips": len(train),
                           "formal": args.formal, "formal_min_count": args.formal_min_count,
                           "min_count": args.min_count, "min_speakers": args.min_speakers},
                 args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("formal", help="standard-Bangla vocabulary")
    f.add_argument("--source", required=True, help="CSV/TSV of standard-Bangla sentences")
    f.add_argument("--column", default="sentence", help="column name, or index if headerless")
    f.add_argument("--min-count", type=int, default=1,
                   help="words stored; the count threshold is applied when the vocab is loaded")
    f.add_argument("--out", default="lexicons/formal_vocab.txt")
    f.set_defaults(func=build_formal)

    def regional(p, out):
        p.add_argument("--formal", default="lexicons/formal_vocab.txt")
        p.add_argument("--formal-min-count", type=int, default=3, help="drops typos and rare forms")
        p.add_argument("--min-count", type=int, default=3)
        p.add_argument("--out", default=out)

    e = sub.add_parser("external", help="regional lexicon from Ben-10 + Vashantor text")
    e.add_argument("--ben10", default="", help="Ben-10 train.csv")
    e.add_argument("--vashantor", default="", help="Vashantor_CSV_Format directory")
    regional(e, "lexicons/external.json")
    e.set_defaults(func=build_external)

    i = sub.add_parser("indomain", help="regional lexicon from train-split ASR transcripts")
    i.add_argument("--transcripts", required=True)
    i.add_argument("--manifest", default="splits/manifest_split_strict_rel.csv")
    i.add_argument("--min-speakers", type=int, default=2)
    regional(i, "lexicons/indomain.json")
    i.set_defaults(func=build_indomain)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
