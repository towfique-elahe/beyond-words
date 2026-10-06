"""
Phase 5 - Dialect ID from ASR transcripts (the lexical route).

Three classifiers over one front-end's transcripts, scored on the same test
clips as the acoustic models (macro-F1 headline, as in Phases 1-3):

  match   Rule-based word matching. Each clip gets a per-region match score:
          the weighted share of its words found in that region's lexicon. A
          clip whose share of non-formal words is under a threshold tuned on
          val, or that matches no lexicon, is called Formal; otherwise the
          best-matching region wins. Interpretable, no training.
  tfidf   TF-IDF (word 1-2 grams + char 2-5 grams) + logistic regression on
          the train transcripts. The learned text baseline.
  fusion  Late fusion with XLS-R: alpha * log p_xlsr + (1 - alpha) * log p_tfidf,
          alpha tuned on val. Needs --xlsr-probs (python predict.py --dump-probs).

A lexicon that covers only some classes (lexicons/external.json) is scored on
the test clips of those classes plus Formal.

Usage:
    python lexical_did.py --transcripts transcripts/tugstugi_ben10.csv \
        --manifest splits/manifest_split_strict_rel.csv \
        --lexicon lexicons/indomain_tugstugi_ben10_strict.json \
        --xlsr-probs reports/phase5/xlsr_probs_strict.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline, make_union

from text_utils import load_vocab, tokenize

FORMAL = "Formal"


# ----------------------------- word matching -----------------------------
def match_scores(tokens, formal, lexicon):
    """Non-formal share of a clip's words, and per region the weighted share of
    its words found in that region's lexicon (the 'match percentage')."""
    n = max(len(tokens), 1)
    nonformal = sum(w not in formal for w in tokens) / n
    scores = {r: sum(words.get(w, 0.0) for w in tokens) / n
              for r, words in lexicon.items() if r != FORMAL}
    return nonformal, scores


def match_predict(nonformal, scores, tau):
    best = max(scores, key=scores.get)
    return FORMAL if nonformal < tau or scores[best] == 0 else best


def run_match(df, formal, lexicon, labels):
    feats = [match_scores(t, formal, lexicon) for t in df["tokens"]]
    val, test = (df["split"] == "val").values, (df["split"] == "test").values

    def predict(tau):
        return np.array([match_predict(nf, sc, tau) for nf, sc in feats])

    taus = np.arange(0.0, 0.52, 0.02)
    f1s = [f1_score(df["district"][val], predict(t)[val], average="macro", labels=labels)
           for t in taus]
    tau = float(taus[int(np.argmax(f1s))])
    covered = np.array([max(sc.values()) > 0 for _, sc in feats])
    return predict(tau), {"tau": round(tau, 2), "coverage_test": round(float(covered[test].mean()), 4)}


# ----------------------------- learned text model -----------------------------
def run_tfidf(df, labels):
    text = df["tokens"].map(" ".join)
    train, val = (df["split"] == "train").values, (df["split"] == "val").values
    best = (-1, None, None)
    for C in [0.3, 1, 3, 10, 30]:
        model = make_pipeline(
            make_union(TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2),
                                       min_df=2, sublinear_tf=True),
                       TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                       min_df=2, sublinear_tf=True)),
            LogisticRegression(C=C, max_iter=3000, class_weight="balanced"))
        model.fit(text[train], df["district"][train])
        f1 = f1_score(df["district"][val], model.predict(text[val]), average="macro")
        if f1 > best[0]:
            best = (f1, C, model)
    _, C, model = best
    assert list(model.classes_) == labels
    return model.predict(text), model.predict_proba(text), {"C": C}


def run_fusion(df, p_lex, xlsr_probs, labels):
    """Late fusion on val/test rows; returns predictions (NaN-free only there)."""
    x = pd.read_csv(xlsr_probs).set_index("relpath")
    held = df["split"].isin(["val", "test"]).values
    p_x = x.loc[df["relpath"][held], labels].values
    lp_x, lp_l = np.log(p_x + 1e-9), np.log(p_lex[held] + 1e-9)
    y, val = df["district"][held].values, (df["split"][held] == "val").values
    lab = np.array(labels)

    alphas = np.round(np.arange(0.0, 1.01, 0.05), 2)
    f1s = [f1_score(y[val], lab[(a * lp_x + (1 - a) * lp_l)[val].argmax(1)], average="macro")
           for a in alphas]
    alpha = float(alphas[int(np.argmax(f1s))])
    pred = df["district"].copy()            # train rows are never scored
    pred[held] = lab[(alpha * lp_x + (1 - alpha) * lp_l).argmax(1)]
    xlsr_pred = df["district"].copy()
    xlsr_pred[held] = lab[p_x.argmax(1)]
    return pred.values, xlsr_pred.values, {"alpha": alpha}


# ----------------------------- main -----------------------------
def score(df, pred, labels):
    """val/test macro-F1 and accuracy over the clips whose true class is in labels."""
    out = {}
    for split in ["val", "test"]:
        m = ((df["split"] == split) & df["district"].isin(labels)).values
        out[f"{split}_macro_f1"] = round(float(f1_score(
            df["district"][m], pred[m], average="macro", labels=labels)), 4)
        out[f"{split}_acc"] = round(float(accuracy_score(df["district"][m], pred[m])), 4)
    m = ((df["split"] == "test") & df["district"].isin(labels)).values
    per = f1_score(df["district"][m], pred[m], average=None, labels=labels)
    out["test_f1_per_class"] = {l: round(float(f), 4) for l, f in zip(labels, per)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--manifest", default="splits/manifest_split_strict_rel.csv")
    ap.add_argument("--formal", default="lexicons/formal_vocab.txt")
    ap.add_argument("--formal-min-count", type=int, default=3)
    ap.add_argument("--lexicon", required=True, help="regional lexicon JSON (build_lexicon.py)")
    ap.add_argument("--xlsr-probs", default="", help="enables late fusion")
    ap.add_argument("--out-dir", default="reports/phase5")
    ap.add_argument("--name", default="", help="run name; default <transcripts>_<lexicon>")
    ap.add_argument("--allow-missing", action="store_true", help="pilot: partial transcripts")
    args = ap.parse_args()

    name = args.name or f"{Path(args.transcripts).stem}__{Path(args.lexicon).stem}"
    man = pd.read_csv(args.manifest)
    df = man.merge(pd.read_csv(args.transcripts)[["relpath", "text"]], on="relpath")
    if len(df) < len(man) and not args.allow_missing:
        raise SystemExit(f"only {len(df)} of {len(man)} clips have a transcript "
                         "(--allow-missing for a pilot)")
    df["tokens"] = df["text"].map(tokenize)
    labels = sorted(man["district"].unique())
    formal = load_vocab(args.formal, args.formal_min_count)
    lexicon = json.load(open(args.lexicon, encoding="utf-8"))["lexicon"]
    lex_labels = sorted(set(lexicon) | {FORMAL})
    print(f"{name}: {len(df)} clips, lexicon covers {len(lex_labels)} of {len(labels)} classes")

    results, preds = {}, {}
    preds["match"], extra = run_match(df, formal, lexicon, lex_labels)
    results["match"] = {**score(df, preds["match"], lex_labels), **extra, "classes": lex_labels}

    preds["tfidf"], p_lex, extra = run_tfidf(df, labels)
    results["tfidf"] = {**score(df, preds["tfidf"], labels), **extra}

    if args.xlsr_probs:
        preds["fusion"], preds["xlsr"], extra = run_fusion(df, p_lex, args.xlsr_probs, labels)
        results["fusion"] = {**score(df, preds["fusion"], labels), **extra}
        results["xlsr"] = score(df, preds["xlsr"], labels)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    test = df[df["split"] == "test"]
    for clf, pred in preds.items():
        if clf == "xlsr":
            continue
        rows = test[test["district"].isin(lex_labels)] if clf == "match" else test
        rows.assign(true=rows["district"], pred=pred[rows.index])[
            ["relpath", "district", "split", "true", "pred"]].to_csv(
            out / f"test_predictions_{name}__{clf}.csv", index=False)
    with open(out / f"metrics_{name}.json", "w", encoding="utf-8") as f:
        json.dump({"transcripts": args.transcripts, "manifest": args.manifest,
                   "lexicon": args.lexicon, "n_clips": len(df), "results": results}, f, indent=2)

    for clf, r in results.items():
        extras = {k: v for k, v in r.items() if k in ("tau", "coverage_test", "C", "alpha")}
        print(f"  {clf:<7} val F1 {r['val_macro_f1']:.4f}   test F1 {r['test_macro_f1']:.4f}   "
              f"test acc {r['test_acc']:.4f}   {extras}")
    print(f"saved -> {out}/metrics_{name}.json")


if __name__ == "__main__":
    main()
