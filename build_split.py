"""
Phase 0 - Split builder.

Reads manifest.csv and writes the same rows back with a new 'split' column
(train/val/test). Strategy:

  * Rows that HAVE a source_id (mostly Formal): grouped split so no source_id
    appears in more than one split  -> honest, source-disjoint.
  * Rows WITHOUT a source_id (the 8 regional classes): stratified clip-level
    split. This CAN be optimistic if the same speaker/source recurs across
    splits. The script prints a clear warning so you disclose it in the thesis.

Later (Phase 3) you can fill manifest 'speaker_id' via clustering and switch
those classes to a grouped split too -- the function already supports it.

Usage:
    python build_split.py --manifest manifest.csv --out manifest_split.csv \
        --val 0.1 --test 0.15 --seed 42
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


def stratified_clip_split(df, val, test, seed):
    """Plain stratified split by district (no grouping)."""
    train_idx, temp_idx = train_test_split(
        df.index, test_size=val + test, random_state=seed,
        stratify=df["district"])
    temp = df.loc[temp_idx]
    rel_test = test / (val + test)
    val_idx, test_idx = train_test_split(
        temp.index, test_size=rel_test, random_state=seed,
        stratify=temp["district"])
    s = pd.Series("train", index=df.index)
    s.loc[val_idx] = "val"
    s.loc[test_idx] = "test"
    return s


def grouped_split(df, val, test, seed):
    """
    Group-aware: a group key (source_id, else speaker_id, else unique per row)
    never crosses splits. Stratified by district as much as grouping allows.
    """
    src = df["source_id"].fillna("").astype(str)
    spk = df.get("speaker_id", pd.Series("", index=df.index)).fillna("").astype(str)
    group = src.where(src != "", spk)
    # rows still without a group get a unique key so they split freely
    group = group.where(group != "", pd.Series(
        ["_solo_" + str(i) for i in range(len(df))], index=df.index))

    n_splits = max(2, int(round(1 / test)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_val_idx, test_idx = next(sgkf.split(df, df["district"], group))
    tv = df.iloc[train_val_idx]

    # carve val out of the train_val portion, still group-aware
    rel_val = val / (1 - test)
    n2 = max(2, int(round(1 / rel_val)))
    sgkf2 = StratifiedGroupKFold(n_splits=n2, shuffle=True, random_state=seed)
    g2 = group.iloc[train_val_idx]
    tr_idx, va_idx = next(sgkf2.split(tv, tv["district"], g2))

    s = pd.Series("train", index=df.index)
    s.iloc[df.index.get_indexer(tv.iloc[va_idx].index)] = "val"
    s.iloc[df.index.get_indexer(df.iloc[test_idx].index)] = "test"
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--out", default="manifest_split.csv")
    ap.add_argument("--val", type=float, default=0.10)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest).reset_index(drop=True)
    df["source_id"] = df["source_id"].fillna("").astype(str)
    df = df[df["status"].isin(["ok", "standardized"])].reset_index(drop=True)

    has_src = df["source_id"].fillna("") != ""
    n_grouped = int(has_src.sum())

    # Decide per-row. Simplest robust approach: if a meaningful fraction has
    # groups, do grouped on the whole frame (solo rows split freely). Otherwise
    # pure stratified clip-level.
    if n_grouped >= 0.05 * len(df):
        df["split"] = grouped_split(df, args.val, args.test, args.seed)
        mode = "grouped (source-disjoint where source_id exists)"
    else:
        df["split"] = stratified_clip_split(df, args.val, args.test, args.seed)
        mode = "stratified clip-level"

    df.to_csv(args.out, index=False)

    print(f"Split mode: {mode}")
    print(pd.crosstab(df["district"], df["split"]))
    print("\nSplit totals:")
    print(df["split"].value_counts())

    regional_unsourced = df[(df["source_id"].fillna("") == "")]
    if len(regional_unsourced):
        print("\n[!] CAVEAT FOR THESIS: "
              f"{len(regional_unsourced)} clips (the regional classes) have no "
              "source/speaker metadata. Their split is clip-level, so the same "
              "speaker/source may appear in train and test. Report test numbers "
              "for these classes as an UPPER BOUND, and add speaker clustering "
              "(Phase 3) for a stricter estimate.")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
