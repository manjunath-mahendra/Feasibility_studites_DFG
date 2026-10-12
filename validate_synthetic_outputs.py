#!/usr/bin/env python3
"""Validate every saved synthetic fold before the common benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_MODELS = [
    "xconvgen", "ctgan", "ctabgan", "cart",
    "tabddpm", "tabdiff", "dpctgan", "dpgan",
]
DISPLAY_NAMES = {
    "xconvgen": "NextConvGeN",
    "ctgan": "CTGAN",
    "ctabgan": "CTAB-GAN",
    "cart": "CART",
    "tabddpm": "TabDDPM",
    "tabdiff": "TabDiff",
    "dpctgan": "DP-CTGAN (ε=5)",
    "dpgan": "DPGAN (ε=5)",
}
TARGET = "diagnosis"
EXPECTED_CLASSES = {1, 2, 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data"))
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/input_validation.csv"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []

    for model in args.models:
        for fold in range(1, 6):
            fold_name = f"fold_{fold:02d}"
            real_path = args.data_root / "folds" / fold_name / "train.csv"
            syn_path = args.synthetic_root / model / fold_name / "synthetic.csv"
            result = {
                "Model": DISPLAY_NAMES.get(model, model),
                "Fold": fold, "Path": str(syn_path),
                "Exists": syn_path.exists(), "Valid": False, "Reason": "",
            }
            try:
                if not syn_path.exists():
                    raise FileNotFoundError("synthetic.csv is missing")
                real = pd.read_csv(real_path)
                syn = pd.read_csv(syn_path)
                result.update({
                    "Rows": len(syn), "Columns": syn.shape[1],
                    "Duplicates": int(syn.duplicated().sum()),
                    "Missing Cells": int(syn.isna().sum().sum()),
                    "Infinite Cells": int(np.isinf(syn.select_dtypes(include=np.number)).sum().sum()),
                })
                if syn.columns.tolist() != real.columns.tolist():
                    raise ValueError("column names/order differ from train.csv")
                if syn.shape != real.shape:
                    raise ValueError(f"shape {syn.shape}, expected {real.shape}")
                if syn.isna().any().any():
                    raise ValueError("contains missing values")
                if np.isinf(syn.select_dtypes(include=np.number)).any().any():
                    raise ValueError("contains infinite values")
                classes = set(pd.to_numeric(syn[TARGET], errors="raise").astype(int).unique())
                result["Class Counts"] = json.dumps(
                    {str(k): int(v) for k, v in syn[TARGET].value_counts().sort_index().items()}
                )
                if classes != EXPECTED_CLASSES:
                    raise ValueError(f"diagnosis classes are {sorted(classes)}")
                result["Valid"] = True
                result["Reason"] = "OK"
            except Exception as error:
                result["Reason"] = str(error)
            rows.append(result)

    report = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.output, index=False)
    print(report.to_string(index=False))
    valid = int(report["Valid"].sum())
    print(f"\nValid model-fold files: {valid}/{len(report)}")
    print("Saved:", args.output)
    return 0 if valid == len(report) else 1


if __name__ == "__main__":
    raise SystemExit(main())
