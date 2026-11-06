"""
Words route of the demo: turn text evidence into one honest distribution.

The thesis classifier (TF-IDF + logistic regression on ASR transcripts, weakly
regularised) gives extreme probabilities on short sentences: one shared word can
reach 99 %. Word lists built from human writing behave the other way round,
strong on typed text and weak on recogniser output. So the demo combines them,
with weights fitted separately for the two kinds of text:

  s(c) = a * log p_tfidf(c)
       + b * sum_w log(p(c|w) + delta)                     (word lists)
       + g * (share of words in the standard vocabulary) * [c == Standard]
  p(c) = softmax(s(c))

a, b and g minimise the negative log-likelihood on validation data, so the
percentages shown are calibrated as well as the ranking being tuned. delta caps
how much a word found in only one region can count, and is chosen the same way.

Rows where the word lists and the classifier disagree are reported separately.
They are already calibrated (a temperature fitted on them came out near 1), so
the demo flags them to the reader instead of rescaling them.

Data (test sets are scored once)
  typed       fit on Vashantor Validation; test on Vashantor Test. Human-written,
              five dialects plus the standard side of the same pairs as Standard.
              Needs an external lexicon built WITHOUT Vashantor Validation/Test
              (--external); the deployed lexicon uses all of Vashantor.
  ben10 text  test only: Ben-10 validation human transcripts (Barishal, Chattogram,
              Sylhet), first 12 words, independent of the Vashantor writing style.
  transcript  fit on strict validation ASR transcripts plus half of the Ben-10
              validation recogniser output (first 12 words, speakers unseen by
              the classifier); test on strict test and on the other half.

With the served model's probabilities (--acoustic) the acoustic temperature and
the fusion weight alpha are tuned on strict validation and scored on strict test.

Writes reports/demo/words_route.json and app/models/words_route.json.

    python tune_words_route.py --external <train-only external lexicon> \
        --acoustic reports/demo/robust_probs_strict.csv
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, f1_score

from lexical_did import lexicon_evidence
from text_utils import load_vocab, tokenize

MODEL = "app/models/lexical_tugstugi_ben10_strict.joblib"
REGION = {"Barishal": "Barishal", "Chittagong": "Chattogram", "Mymensingh": "Mymensingh",
          "Noakhali": "Noakhali", "Sylhet": "Sylhet"}
BEN10 = {"barishal": "Barishal", "chittagong": "Chattogram", "sylhet": "Sylhet"}
DELTA = 0.02            # placeholder until use_delta() sets the fitted value
DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
FORMAL = "Formal"


def vashantor(split, root="data/vashantor/Vashantor_CSV_Format"):
    rows, seen = [], set()
    for path in sorted(Path(root, split).glob("*.csv")):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        col = next(c for c in df.columns if c.endswith("_bangla_speech") and c != "bangla_speech")
        region = REGION[path.name.split()[0]]
        for standard, regional in zip(df["bangla_speech"], df[col]):
            rows.append((regional, region))
            if standard not in seen and len(seen) % 5 == 0:   # about one region's worth of Standard
                rows.append((standard, FORMAL))
            seen.add(standard)
    return pd.DataFrame(rows, columns=["text", "district"])


def ben10_human(words=12):
    df = pd.read_csv("data/ben10/valid.csv")
    df = df[df.district.isin(BEN10)]
    return pd.DataFrame({"text": [" ".join(tokenize(t)[:words]) for t in df.transcripts],
                         "district": df.district.map(BEN10).values})


def ben10_asr(half, words=12, path="transcripts/ben10_valid/tugstugi_ben10.csv"):
    """Recogniser output on Ben-10 validation speech; half 0 for fitting, 1 for testing."""
    df = pd.read_csv(path)
    df = df[df.district.isin(BEN10)].sort_values("relpath").reset_index(drop=True)
    df = df[df.index % 2 == half]
    return pd.DataFrame({"text": [" ".join(tokenize(t)[:words]) for t in df.text],
                         "district": df.district.map(BEN10).values})


class Features:
    def __init__(self, df, pipe, lexicons, standard, classes):
        toks = df["text"].map(tokenize)
        keep = toks.map(len).values > 0
        self.y, toks = df["district"].values[keep], toks[keep]
        self.relpath = df["relpath"].values[keep] if "relpath" in df else None
        self.logp = np.log(pipe.predict_proba(toks.map(" ".join)) + 1e-9)
        self.lex = np.stack([lexicon_evidence(t, lexicons, classes, DELTA) for t in toks])
        self.std = np.array([np.mean([w in standard for w in t]) for t in toks])
        self.n = toks.map(len).values
        self.toks, self.lexicons, self.classes = toks, lexicons, classes

    def use_delta(self, delta):
        self.lex = np.stack([lexicon_evidence(t, self.lexicons, self.classes, delta) for t in self.toks])


def logits(f, classes, a, b, g):
    s = a * f.logp + b * f.lex
    s[:, classes.index(FORMAL)] += g * f.std
    return s


def softmax(s):
    z = np.exp(s - s.max(1, keepdims=True))
    return z / z.sum(1, keepdims=True)


def conflict(f):
    """Rows with a word-list hit whose best class differs from the classifier's."""
    return f.lex.any(axis=1) & (f.lex.argmax(1) != f.logp.argmax(1))


def fit_temperature(s, y, classes):
    idx = np.array([classes.index(c) for c in y])
    temps = np.geomspace(0.5, 32, 60)
    nll = [-np.log(softmax(s / t)[np.arange(len(y)), idx] + 1e-12).mean() for t in temps]
    return float(temps[int(np.argmin(nll))])


def fit_weights(sets, classes):
    """a, b, g >= 0 minimising the mean negative log-likelihood, each set counted equally."""
    def nll(w):
        total = 0.0
        for f in sets:
            s = logits(f, classes, *w)
            s = s - s.max(1, keepdims=True)
            idx = np.array([classes.index(c) for c in f.y])
            total += (np.log(np.exp(s).sum(1)) - s[np.arange(len(idx)), idx]).mean()
        return total / len(sets)
    r = minimize(nll, x0=[.5, .5, .5], bounds=[(0, 5)] * 3, method="L-BFGS-B")
    return [round(float(v), 3) for v in r.x], float(r.fun)


def report(y, p, classes, labels=None):
    labels = labels or sorted(set(y))
    pred = np.array(classes)[p.argmax(1)]
    top = p.max(1)
    return {"macro_f1": round(float(f1_score(y, pred, average="macro", labels=labels)), 3),
            "accuracy": round(float(accuracy_score(y, pred)), 3),
            "mean_confidence": round(float(top.mean()), 3),
            "share_over_95": round(float((top > .95).mean()), 3),
            "accuracy_over_95": round(float((pred == y)[top > .95].mean()), 3) if (top > .95).any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True, help="external lexicon built without Vashantor Validation/Test")
    ap.add_argument("--indomain", default="lexicons/indomain_tugstugi_ben10_strict.json")
    ap.add_argument("--acoustic", default="", help="val/test probabilities of the served acoustic model")
    ap.add_argument("--out", default="reports/demo/words_route.json")
    ap.add_argument("--model-out", default="app/models/words_route.json")
    args = ap.parse_args()

    bundle = joblib.load(MODEL)
    pipe, classes = bundle["pipeline"], list(bundle["classes"])
    standard = load_vocab("lexicons/formal_vocab.txt", 3)
    lexicons = [json.load(open(args.external, encoding="utf-8"))["lexicon"],
                json.load(open(args.indomain, encoding="utf-8"))["lexicon"]]
    man = pd.read_csv("splits/manifest_split_strict_rel.csv")
    corpus = man.merge(pd.read_csv("transcripts/tugstugi_ben10.csv")[["relpath", "text"]], on="relpath")
    data = {"transcript_val": corpus[corpus.split == "val"], "transcript_test": corpus[corpus.split == "test"],
            "typed_val": vashantor("Validation"), "typed_test": vashantor("Test"),
            "ben10_text_test": ben10_human(), "ben10_asr_val": ben10_asr(0), "ben10_asr_test": ben10_asr(1)}
    F = {k: Features(v, pipe, lexicons, standard, classes) for k, v in data.items()}
    print({k: len(f.y) for k, f in F.items()})

    settings, results = {}, {}
    routes = (("typed", ["typed_val"], ["typed_test", "ben10_text_test"]),
              ("transcript", ["transcript_val", "ben10_asr_val"], ["transcript_test", "ben10_asr_test"]))
    for route, fit_on, tests in routes:
        best = None
        for delta in DELTAS:
            for n in fit_on:
                F[n].use_delta(delta)
            w, nll = fit_weights([F[n] for n in fit_on], classes)
            print(f"  {route} delta {delta}: a {w[0]} b {w[1]} g {w[2]} val NLL {nll:.4f}")
            if best is None or nll < best[0]:
                best = (nll, delta, w)
        _, delta, (a, b, g) = best
        for n in fit_on:
            F[n].use_delta(delta)
        settings[route] = {"a": a, "b": b, "g": g, "delta": delta}
        print(f"{route}: a {a} b {b} g {g} delta {delta}")
        for t in tests:
            f = F[t]
            f.use_delta(delta)
            labels = sorted(set(f.y))
            results[f"{t} / classifier (old)"] = report(f.y, softmax(f.logp), classes, labels)
            results[f"{t} / word lists only"] = report(f.y, softmax(f.lex), classes, labels)
            s = logits(f, classes, a, b, g)
            c = conflict(f)
            results[f"{t} / disagreement rows"] = report(f.y[c], softmax(s[c]), classes, labels)
            results[f"{t} / new"] = report(f.y, softmax(s), classes, labels)
            short = f.n <= 6
            if short.sum() >= 30:
                results[f"{t} / new, up to 6 words"] = report(f.y[short], softmax(s)[short], classes, labels)
                results[f"{t} / classifier (old), up to 6 words"] = report(f.y[short], softmax(f.logp)[short], classes, labels)

    if args.acoustic:
        a = pd.read_csv(args.acoustic).set_index("relpath")
        pa = {k: a.loc[F[f"transcript_{k}"].relpath, classes].values for k in ("val", "test")}
        val, test = F["transcript_val"], F["transcript_test"]
        la = {k: np.log(v + 1e-9) for k, v in pa.items()}
        ta = fit_temperature(la["val"], val.y, classes)
        t = settings["transcript"]
        lw = {"val": np.log(softmax(logits(val, classes, t["a"], t["b"], t["g"])) + 1e-9),
              "test": np.log(softmax(logits(test, classes, t["a"], t["b"], t["g"])) + 1e-9)}
        alphas = np.round(np.arange(0, 1.01, .05), 2)
        def fused(k, al):
            return softmax(al * la[k] / ta + (1 - al) * lw[k])
        alpha = float(max(alphas, key=lambda al: report(val.y, fused("val", al), classes)["macro_f1"]))
        settings["acoustic_temperature"] = round(ta, 3)
        settings["alpha"] = alpha
        print(f"acoustic temperature {ta:.2f}, alpha {alpha}")
        results["sound alone, strict test"] = report(test.y, softmax(la["test"] / ta), classes)
        results["sound + transcript, strict test"] = report(test.y, fused("test", alpha), classes)
        results["sound + transcript, alpha 0.25, strict test"] = report(test.y, fused("test", .25), classes)

    for k, v in results.items():
        print(f"{k:48s} F1 {v['macro_f1']:.3f} acc {v['accuracy']:.3f} conf {v['mean_confidence']:.3f} >95%: {v['share_over_95']:.2f} right {v['accuracy_over_95']}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"settings": settings, "sizes": {k: len(f.y) for k, f in F.items()}, "results": results}, open(args.out, "w"), indent=2)
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(settings, open(args.model_out, "w"), indent=2)
    print("wrote", args.out, "and", args.model_out)


if __name__ == "__main__":
    main()
