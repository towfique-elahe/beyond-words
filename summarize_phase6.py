"""
Phase 6 - One table over every acoustic model: pretraining x regime x split.

Collects test macro-F1 (and per-class F1) from
    reports/phase2/metrics_xlsr.json, reports/phase3/metrics_xlsr_strict.json   (Phase 2/3)
    reports/phase6/{clip,strict}/metrics_<tag>.json                             (Phase 6 runs,
                                                frozen_* rows from baseline_whisper.py train)
and writes reports/phase6/summary.csv and summary_per_class.csv.

Usage:
    python summarize_phase6.py
"""

import json
from pathlib import Path

import pandas as pd

R = Path("reports")
# tag -> (display name, pretraining, regime)
ROWS = {
    "mfcc":                  ("MFCC statistics + MLP",        "none",          "hand-crafted"),
    "frozen_whisper_small":  ("Whisper-small, frozen + MLP",  "multilingual",  "frozen"),
    "frozen_whisper_medium": ("Whisper-medium, frozen + MLP", "multilingual",  "frozen"),
    "frozen_tugstugi_ben10": ("Tugstugi (Ben-10), frozen + MLP", "Bangla",     "frozen"),
    "xlsr":                  ("XLS-R-300M, fine-tuned",       "multilingual",  "fine-tuned"),
    "whisper_medium":        ("Whisper-medium, fine-tuned",   "multilingual",  "fine-tuned"),
    "tugstugi_ben10":        ("Tugstugi (Ben-10), fine-tuned", "Bangla",       "fine-tuned"),
    "tugstugi":              ("Tugstugi, fine-tuned",         "Bangla",        "fine-tuned"),
    "hishab":                ("Hishab FastConformer, fine-tuned", "Bangla",    "fine-tuned"),
}
LEGACY = {  # (tag, split) -> path, for rows produced before Phase 6
    ("xlsr", "clip"): R / "phase2/metrics_xlsr.json",
    ("xlsr", "strict"): R / "phase3/metrics_xlsr_strict.json",
}


def load_metrics(path):
    m = json.load(open(path))
    return m["test_macroF1"], m["per_class_F1"]


def main():
    rows, per_class = [], []
    for tag, (name, pretraining, regime) in ROWS.items():
        rec = {"tag": tag, "model": name, "pretraining": pretraining, "regime": regime}
        for split in ["clip", "strict"]:
            path = LEGACY.get((tag, split), R / f"phase6/{split}/metrics_{tag}.json")
            if not path.exists():
                continue
            macro, per = load_metrics(path)
            rec[f"{split}_f1"] = round(macro, 4)
            per_class.append({"tag": tag, "model": name, "split": split, **per})
        if "clip_f1" in rec and "strict_f1" in rec:
            rec["drop"] = round(rec["clip_f1"] - rec["strict_f1"], 4)
        if len(rec) > 4:
            rows.append(rec)
    df = pd.DataFrame(rows)
    out = R / "phase6"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "summary.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "summary_per_class.csv", index=False)
    print(df.to_string(index=False))
    print(f"saved -> {out}/summary.csv, summary_per_class.csv")


if __name__ == "__main__":
    main()
