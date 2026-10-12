#!/usr/bin/env python3
"""Generate size-matched CTGAN data for the five fibrosis benchmark folds.

The script trains exclusively on each fold's train.csv. It does not read the
held-out validation data. It is designed for SDV 1.38.x and saves one CSV plus
one generation metadata JSON file per fold.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


TARGET = "diagnosis"
EXPECTED_CLASSES = [1, 2, 3]
BASE_SEED = 42

NOMINAL_COLUMNS = [
    "sex",
    "bariatricSurgery",
]

ORDINAL_COLUMNS = [
    "DiabetesMellitusII",
    "ArterielleHypertonie",
    "Hyperlipidämie",
    "Hypertriglycerinämie",
    "KHK",
    TARGET,
]

CATEGORICAL_COLUMNS = NOMINAL_COLUMNS + ORDINAL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CTGAN data for saved fibrosis benchmark folds."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("fibrosis_benchmark_data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("synthetic_data/ctgan"),
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def create_metadata(train: pd.DataFrame) -> SingleTableMetadata:
    missing = [column for column in CATEGORICAL_COLUMNS if column not in train]
    if missing:
        raise ValueError(f"Required categorical columns are missing: {missing}")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=train)

    for column in CATEGORICAL_COLUMNS:
        metadata.update_column(column_name=column, sdtype="categorical")

    metadata.validate()
    metadata.validate_data(data=train)
    return metadata


def clean_synthetic(raw: pd.DataFrame, train: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw, columns=train.columns)

    missing = sorted(set(train.columns) - set(raw.columns))
    if missing:
        raise ValueError(f"CTGAN output is missing columns: {missing}")

    synthetic = raw.loc[:, train.columns].copy()
    for column in synthetic.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")

    synthetic = (
        synthetic.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    for column in CATEGORICAL_COLUMNS:
        synthetic[column] = synthetic[column].round().astype(int)

    synthetic = synthetic[
        synthetic[TARGET].isin(EXPECTED_CLASSES)
    ].reset_index(drop=True)
    return synthetic


def validate_output(synthetic: pd.DataFrame, train: pd.DataFrame) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic column order does not match train.csv")
    if len(synthetic) != len(train):
        raise ValueError(
            f"Expected {len(train)} valid rows but obtained {len(synthetic)}"
        )
    if synthetic.isna().any().any():
        raise ValueError("Synthetic data contain missing values")

    observed_classes = sorted(synthetic[TARGET].unique().astype(int).tolist())
    if observed_classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Generated diagnosis classes are {observed_classes}; "
            f"expected {EXPECTED_CLASSES}"
        )


def generate_fold(
    fold_number: int,
    args: argparse.Namespace,
    use_cuda: bool,
) -> dict:
    fold_name = f"fold_{fold_number:02d}"
    train_path = args.data_root / "folds" / fold_name / "train.csv"
    output_dir = args.output_root / fold_name
    output_path = output_dir / "synthetic.csv"
    metadata_path = output_dir / "generation_metadata.json"
    seed = args.base_seed + fold_number

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists; use --overwrite to replace it"
        )
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    started_at = datetime.now(timezone.utc)
    started_timer = time.perf_counter()
    metadata_record = {
        "model": "CTGAN",
        "fold": fold_number,
        "seed": seed,
        "status": "started",
        "started_at_utc": started_at.isoformat(),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "cuda": use_cuda,
            "enforce_min_max_values": False,
            "enforce_rounding": False,
        },
    }

    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        if TARGET not in train:
            raise ValueError(f"Target column {TARGET!r} is absent")
        if train.isna().any().any():
            raise ValueError("Training data contain missing values")

        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
        real_classes = sorted(train[TARGET].unique().tolist())
        if real_classes != EXPECTED_CLASSES:
            raise ValueError(
                f"Training classes are {real_classes}; expected {EXPECTED_CLASSES}"
            )

        metadata = create_metadata(train)
        synthesizer = CTGANSynthesizer(
            metadata=metadata,
            enforce_min_max_values=False,
            enforce_rounding=False,
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True,
            cuda=use_cuda,
        )

        set_random_state = getattr(synthesizer, "set_random_state", None)
        if callable(set_random_state):
            set_random_state(seed)

        synthesizer.fit(train)
        raw_synthetic = synthesizer.sample(num_rows=len(train))
        raw_rows = len(raw_synthetic)
        synthetic = clean_synthetic(raw_synthetic, train)
        validate_output(synthetic, train)
        synthetic.to_csv(output_path, index=False)

        metadata_record.update(
            {
                "status": "success",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started_timer,
                "requested_rows": len(train),
                "raw_generated_rows": raw_rows,
                "generated_rows": len(synthetic),
                "column_count": len(synthetic.columns),
                "categorical_columns": CATEGORICAL_COLUMNS,
                "continuous_columns": [
                    column
                    for column in train.columns
                    if column not in CATEGORICAL_COLUMNS
                ],
                "training_class_counts": train[TARGET]
                .value_counts()
                .sort_index()
                .to_dict(),
                "synthetic_class_counts": synthetic[TARGET]
                .value_counts()
                .sort_index()
                .to_dict(),
                "synthetic_duplicate_count": int(
                    len(synthetic) - len(synthetic.drop_duplicates())
                ),
                "synthetic_sha256": sha256_file(output_path),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "sdv": package_version("sdv"),
                    "ctgan": package_version("ctgan"),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                    "numpy": package_version("numpy"),
                    "pandas": package_version("pandas"),
                },
            }
        )
    except Exception as error:
        metadata_record.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started_timer,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if output_path.exists():
            output_path.unlink()
    finally:
        metadata_path.write_text(
            json.dumps(json_safe(metadata_record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return metadata_record


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.batch_size % 2 != 0:
        raise ValueError("--batch-size must be a positive even integer")

    use_cuda = torch.cuda.is_available() and not args.cpu
    folds = [args.fold] if args.fold is not None else list(range(1, 6))
    summaries = []

    print("=" * 80)
    print("CTGAN SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("SDV version:", package_version("sdv"))
    print("PyTorch version:", torch.__version__)
    print("CUDA enabled:", use_cuda)
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", args.epochs)
    print("Data root:", args.data_root.resolve())
    print("Output root:", args.output_root.resolve())

    for fold_number in folds:
        print(f"\n{'-' * 80}\nFOLD {fold_number}/5\n{'-' * 80}")
        try:
            result = generate_fold(fold_number, args, use_cuda)
        except Exception as error:
            result = {
                "fold": fold_number,
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        summaries.append(result)
        print("Status:", result["status"].upper())
        if result["status"] == "success":
            print("Rows:", result["generated_rows"])
            print("Classes:", result["synthetic_class_counts"])
            print("Duplicates:", result["synthetic_duplicate_count"])
            print("Seconds:", round(result["elapsed_seconds"], 2))
        else:
            print("Reason:", result.get("error_message", "unknown error"))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    successful = sum(item["status"] == "success" for item in summaries)
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
