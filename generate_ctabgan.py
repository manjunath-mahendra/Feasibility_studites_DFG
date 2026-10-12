#!/usr/bin/env python3
"""Generate size-matched CTAB-GAN data for five fibrosis benchmark folds.

Requires the local ``model`` package. The wrapper trains on each complete
train.csv and never reads validation.csv. It records failures per fold.
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
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from model.ctabgan import CTABGAN
import model.pipeline.data_preparation as data_preparation_module


TARGET = "diagnosis"
EXPECTED_CLASSES = [1, 2, 3]
BASE_SEED = 42

NOMINAL_COLUMNS = ["sex", "bariatricSurgery"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=Path("fibrosis_benchmark_data")
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("synthetic_data/ctabgan")
    )
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
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


def version(name: str) -> str | None:
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


def fit_on_complete_fold(generator: CTABGAN) -> None:
    """Make DataPrep's split a no-op only for this fit call.

    The local DataPrep delegates to sklearn's train_test_split. CTAB-GAN's
    default test_ratio=0.2 is inappropriate here because benchmark validation
    is already stored separately. This temporary replacement returns the full
    input as training data and an empty internal test frame.
    """
    model_selection = data_preparation_module.model_selection
    original_split = model_selection.train_test_split

    def complete_fold_split(features, labels, *args, **kwargs):
        return (
            features.copy(),
            features.iloc[0:0].copy(),
            labels.copy(),
            labels.iloc[0:0].copy(),
        )

    model_selection.train_test_split = complete_fold_split
    try:
        generator.fit()
    finally:
        model_selection.train_test_split = original_split


def clean_output(raw, train: pd.DataFrame, seed: int) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw, columns=train.columns)

    missing = sorted(set(train.columns) - set(raw.columns))
    if missing:
        raise ValueError(f"CTAB-GAN output is missing columns: {missing}")

    synthetic = raw.loc[:, train.columns].copy()
    if len(synthetic) < len(train):
        raise ValueError(
            f"CTAB-GAN returned only {len(synthetic)}/{len(train)} rows"
        )
    if len(synthetic) > len(train):
        synthetic = synthetic.sample(
            n=len(train), replace=False, random_state=seed
        ).reset_index(drop=True)

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
        raise ValueError("Synthetic columns do not match train.csv")
    if len(synthetic) != len(train):
        raise ValueError(
            f"Expected {len(train)} valid rows; obtained {len(synthetic)}"
        )
    classes = sorted(synthetic[TARGET].astype(int).unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Generated classes are {classes}; expected {EXPECTED_CLASSES}"
        )
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
        raise FileExistsError(
            f"{output_path} exists; use --overwrite to replace it"
        )
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    started = time.perf_counter()
    record = {
        "model": "CTAB-GAN",
        "fold": fold_number,
        "seed": seed,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "test_ratio": 0.0,
            "problem_type": {"Classification": TARGET},
            "categorical_columns": CATEGORICAL_COLUMNS,
            "integer_columns": [],
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

        with tempfile.TemporaryDirectory(prefix=f"ctabgan_{fold_name}_") as temp:
            temporary_csv = Path(temp) / "train.csv"
            train.to_csv(temporary_csv, index=False)
            generator = CTABGAN(
                raw_csv_path=str(temporary_csv),
                test_ratio=0.0,
                categorical_columns=CATEGORICAL_COLUMNS,
                log_columns=[],
                mixed_columns={},
                general_columns=[],
                non_categorical_columns=[],
                integer_columns=[],
                problem_type={"Classification": TARGET},
            )

            if hasattr(generator.synthesizer, "epochs"):
                generator.synthesizer.epochs = args.epochs
            if hasattr(generator.synthesizer, "batch_size"):
                generator.synthesizer.batch_size = args.batch_size

            fit_on_complete_fold(generator)
            raw = generator.generate_samples()

        raw_rows = len(raw)
        synthetic = clean_output(raw, train, seed)
        validate_output(synthetic, train)
        synthetic.to_csv(output_path, index=False)

        record.update(
            {
                "status": "success",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "requested_rows": len(train),
                "raw_generated_rows": raw_rows,
                "generated_rows": len(synthetic),
                "training_class_counts": train[TARGET]
                .value_counts().sort_index().to_dict(),
                "synthetic_class_counts": synthetic[TARGET]
                .value_counts().sort_index().to_dict(),
                "synthetic_duplicate_count": int(synthetic.duplicated().sum()),
                "synthetic_sha256": sha256_file(output_path),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "cuda_available": torch.cuda.is_available(),
                    "gpu": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available() else None
                    ),
                    "numpy": version("numpy"),
                    "pandas": version("pandas"),
                    "scikit_learn": version("scikit-learn"),
                },
            }
        )
    except Exception as error:
        record.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - started,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if output_path.exists():
            output_path.unlink()
    finally:
        metadata_path.write_text(
            json.dumps(json_safe(record), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return record


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0 or args.batch_size % 2:
        raise ValueError("--batch-size must be a positive even integer")

    folds = [args.fold] if args.fold else list(range(1, 6))
    results = []
    print("=" * 80)
    print("CTAB-GAN SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print("Epochs:", args.epochs)
    print("Data root:", args.data_root.resolve())
    print("Output root:", args.output_root.resolve())

    for fold in folds:
        print(f"\n{'-' * 80}\nFOLD {fold}/5\n{'-' * 80}")
        try:
            result = generate_fold(fold, args)
        except Exception as error:
            result = {
                "fold": fold,
                "status": "failed",
                "error_message": str(error),
            }
        results.append(result)
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

    successful = sum(result["status"] == "success" for result in results)
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
