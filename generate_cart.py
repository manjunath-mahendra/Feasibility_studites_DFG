#!/usr/bin/env python3
"""Generate size-matched CART synthetic data for fibrosis benchmark folds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from synthpop import CARTMethod, DataProcessor, MissingDataHandler


TARGET = "diagnosis"
EXPECTED_CLASSES = [1, 2, 3]
BASE_SEED = 42
CATEGORICAL_COLUMNS = [
    "sex",
    "bariatricSurgery",
    "DiabetesMellitusII",
    "ArterielleHypertonie",
    "Hyperlipidämie",
    "Hypertriglycerinämie",
    "KHK",
    TARGET,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/cart"))
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--minibucket", type=int, default=5)
    parser.add_argument("--smoothing", action="store_true")
    parser.add_argument("--proper", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def clean_output(raw: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw, columns=train.columns)
    missing = sorted(set(train.columns) - set(raw.columns))
    if missing:
        raise ValueError(f"CART output is missing columns: {missing}")

    synthetic = raw.loc[:, train.columns].copy()
    for column in train.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    for column in CATEGORICAL_COLUMNS:
        synthetic[column] = synthetic[column].round().astype(int)
    synthetic = synthetic[synthetic[TARGET].isin(EXPECTED_CLASSES)].reset_index(drop=True)
    return synthetic


def validate_output(synthetic: pd.DataFrame, train: pd.DataFrame) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic columns do not match train.csv")
    if len(synthetic) != len(train):
        raise ValueError(f"Expected {len(train)} valid rows; obtained {len(synthetic)}")
    classes = sorted(synthetic[TARGET].unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Generated classes are {classes}; expected {EXPECTED_CLASSES}")
    if synthetic.isna().any().any():
        raise ValueError("Synthetic data contain missing values")


def generate_fold(fold_number: int, args: argparse.Namespace) -> dict:
    fold_name = f"fold_{fold_number:02d}"
    train_path = args.data_root / "folds" / fold_name / "train.csv"
    output_dir = args.output_root / fold_name
    output_path = output_dir / "synthetic.csv"
    metadata_path = output_dir / "generation_metadata.json"
    seed = args.base_seed + fold_number

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite to replace it")
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    started = time.perf_counter()
    record = {
        "model": "CART",
        "fold": fold_number,
        "seed": seed,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "num_rows": 104,
            "minibucket": args.minibucket,
            "smoothing": args.smoothing,
            "proper": args.proper,
            "enforce_rounding": True,
            "enforce_min_max_values": True,
            "categorical_columns": CATEGORICAL_COLUMNS,
        },
    }

    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        if train.isna().any().any():
            raise ValueError("Training data contain missing values")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
        if sorted(train[TARGET].unique().tolist()) != EXPECTED_CLASSES:
            raise ValueError("Training fold does not contain all target classes")

        handler = MissingDataHandler()
        metadata = handler.get_column_dtypes(train)
        for column in CATEGORICAL_COLUMNS:
            if column not in metadata:
                raise ValueError(f"Required categorical column is missing: {column}")
            metadata[column] = "categorical"
        for column in train.columns:
            if column not in CATEGORICAL_COLUMNS:
                metadata[column] = "numerical"

        processor = DataProcessor(
            metadata,
            enforce_rounding=True,
            enforce_min_max_values=True,
        )
        processed = processor.preprocess(train)

        cart = CARTMethod(
            metadata,
            smoothing=args.smoothing,
            proper=args.proper,
            minibucket=args.minibucket,
            random_state=seed,
        )
        cart.fit(processed)
        synthetic_processed = cart.sample(len(train))
        raw = processor.postprocess(synthetic_processed)

        synthetic = clean_output(raw, train)
        validate_output(synthetic, train)
        synthetic.to_csv(output_path, index=False)

        record.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "requested_rows": len(train),
            "generated_rows": len(synthetic),
            "training_class_counts": train[TARGET].value_counts().sort_index().to_dict(),
            "synthetic_class_counts": synthetic[TARGET].value_counts().sort_index().to_dict(),
            "synthetic_duplicates": int(synthetic.duplicated().sum()),
            "metadata": metadata,
        })
        print("Status: SUCCESS")
        print("Rows:", len(synthetic))
        print("Classes:", synthetic[TARGET].value_counts().sort_index().to_dict())
        print("Duplicates:", int(synthetic.duplicated().sum()))
        print("Seconds:", round(record["elapsed_seconds"], 2))

    except Exception as error:
        record.update({
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        print("Status: FAILED")
        print("Reason:", error)

    record["software"] = {
        "python": sys.version,
        "platform": platform.platform(),
        "python-synthpop": package_version("python-synthpop"),
        "pandas": package_version("pandas"),
        "numpy": package_version("numpy"),
        "scikit-learn": package_version("scikit-learn"),
    }
    metadata_path.write_text(
        json.dumps(json_safe(record), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return record


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    folds = [args.fold] if args.fold else list(range(1, 6))

    print("=" * 80)
    print("CART SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("Data root:", args.data_root)
    print("Output root:", args.output_root)
    print("Smoothing:", args.smoothing)
    print("Proper:", args.proper)

    results = []
    for fold in folds:
        print("\n" + "-" * 80)
        print(f"FOLD {fold}/5")
        print("-" * 80)
        results.append(generate_fold(fold, args))

    successes = sum(item["status"] == "success" for item in results)
    print(f"\nSuccessful folds: {successes}/{len(results)}")
    if successes != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
