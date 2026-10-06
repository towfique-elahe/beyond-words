"""
Phase 5 - One table from the per-run metrics in reports/phase5/.

Rows: ASR front-end x split. Columns: the front-end diagnostics on the thesis
clips (mean OOV share on regional classes vs Formal) and the test macro-F1 of
the three lexical classifiers, next to XLS-R on the same split.

Usage:
    python summarize_phase5.py            # prints and writes reports/phase5/summary.csv
"""

import json
from pathlib import Path

import pandas as pd

R = Path("reports/phase5")
FRONTS = ["tugstugi_ben10", "tugstugi", "whisper_large_v3", "w2v_xlsr_bn", "hishab_fastconformer"]


def main():
    rows = []
    for front in FRONTS:
        diag = json.load(open(R / f"diagnostics_{front}.json", encoding="utf-8"))
        table = pd.DataFrame(diag["fronts"][front]["table"]).set_index("district")
        oov_formal = table.loc["Formal", "oov"]
        oov_regional = table.drop(index="Formal")["oov"].mean()
        for split in ["strict", "clip"]:
            path = R / f"metrics_{front}__indomain_{front}_{split}.json"
            if not path.exists():
                continue
            res = json.load(open(path))["results"]
            rows.append({
                "front_end": front, "split": split,
                "oov_formal": round(oov_formal, 3), "oov_regional": round(oov_regional, 3),
                "match_f1": res["match"]["test_macro_f1"],
                "match_coverage": res["match"]["coverage_test"],
                "tfidf_f1": res["tfidf"]["test_macro_f1"],
                "fusion_f1": res["fusion"]["test_macro_f1"], "alpha": res["fusion"]["alpha"],
                "xlsr_f1": res["xlsr"]["test_macro_f1"],
            })
    df = pd.DataFrame(rows).sort_values(["split", "fusion_f1"], ascending=[False, False])
    print(df.to_string(index=False))
    df.to_csv(R / "summary.csv", index=False)
    print(f"saved -> {R / 'summary.csv'}")


if __name__ == "__main__":
    main()
