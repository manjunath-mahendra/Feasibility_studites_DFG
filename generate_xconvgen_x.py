#!/usr/bin/env python3
"""Generate size-matched XConvGeN data for all saved fibrosis folds.

Run this script from the Feasibility_studies_dfg repository root. It trains
XConvGeN only on each fold's train.csv and never reads validation.csv.
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

from fdc.fdc import FDC
from library.generators.XConvGeN import GeneratorConfig, XConvGeN


TARGET = "diagnosis"
BASE_SEED = 42
EXPECTED_CLASSES = [1, 2, 3]

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("fibrosis_benchmark_data"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("synthetic_data/xconvgen"),
    )
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except Exception:
        pass


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
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def build_fdc(train: pd.DataFrame) -> tuple[FDC, list[str]]:
    missing = [
        column
        for column in NOMINAL_COLUMNS + ORDINAL_COLUMNS
        if column not in train.columns
    ]
    if missing:
        raise ValueError(f"Required typed columns are missing: {missing}")

    continuous_columns = [
        column
        for column in train.columns
        if column not in NOMINAL_COLUMNS + ORDINAL_COLUMNS
    ]

    fdc = FDC()
    fdc.nom_list = [train.columns.get_loc(column) for column in NOMINAL_COLUMNS]
    fdc.ord_list = [train.columns.get_loc(column) for column in ORDINAL_COLUMNS]
    fdc.cont_list = [
        train.columns.get_loc(column) for column in continuous_columns
    ]
    return fdc, continuous_columns


def clean_synthetic(
    raw_synthetic,
    train: pd.DataFrame,
) -> pd.DataFrame:
    synthetic = pd.DataFrame(raw_synthetic, columns=train.columns)

    for column in synthetic.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")

    synthetic = (
        synthetic.replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    for column in NOMINAL_COLUMNS + ORDINAL_COLUMNS:
        synthetic[column] = synthetic[column].round().astype(int)

    synthetic = synthetic[
        synthetic[TARGET].isin(EXPECTED_CLASSES)
    ].reset_index(drop=True)
    return synthetic


def validate_output(synthetic: pd.DataFrame, train: pd.DataFrame) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic columns do not exactly match training columns")
    if len(synthetic) != len(train):
        raise ValueError(
            f"Expected {len(train)} valid rows, obtained {len(synthetic)}"
        )
    if synthetic.isna().any().any():
        raise ValueError("Synthetic data contain missing values")
    if np.isinf(synthetic.select_dtypes(include=[np.number])).any().any():
        raise ValueError("Synthetic data contain infinite values")

    observed_classes = sorted(
        synthetic[TARGET].astype(int).unique().tolist()
    )
    if observed_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Synthetic diagnosis classes are "
            f"{observed_classes}; expected {EXPECTED_CLASSES}"
        )


def generate_fold(
    fold_number: int,
    data_root: Path,
    output_root: Path,
    base_seed: int,
    overwrite: bool,
) -> dict:
    fold_name = f"fold_{fold_number:02d}"
    train_path = data_root / "folds" / fold_name / "train.csv"
    output_dir = output_root / fold_name
    output_path = output_dir / "synthetic.csv"
    metadata_path = output_dir / "generation_metadata.json"
    seed = base_seed + fold_number

    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace it."
        )
    if not train_path.exists():
        raise FileNotFoundError(train_path)

    started_at = datetime.now(timezone.utc)
    start_time = time.perf_counter()
    metadata = {
        "model": "XConvGeN",
        "fold": fold_number,
        "seed": seed,
        "status": "started",
        "started_at_utc": started_at.isoformat(),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "neb": 5,
            "genAddNoise": False,
            "alpha_clip": 0.1,
        },
    }

    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)

        if TARGET not in train.columns:
            raise ValueError(f"Target column {TARGET!r} is absent")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)

        real_classes = sorted(train[TARGET].unique().tolist())
        if real_classes != EXPECTED_CLASSES:
            raise ValueError(
                f"Training classes are {real_classes}; expected {EXPECTED_CLASSES}"
            )

        if train.isna().any().any():
            raise ValueError(
                "XConvGeN input contains missing values. Do not impute inside this "
                "script because that would change the shared benchmark data."
            )

        fdc, continuous_columns = build_fdc(train)
        training_array = train.to_numpy()
        n_required = len(train)

        config = GeneratorConfig(
            n_feat=training_array.shape[1],
            neb=5,
            genAddNoise=False,
            alpha_clip=0.1,
        )
        generator = XConvGeN(config=config, fdc=fdc, debug=False)
        generator.reset(training_array)
        generator.train(training_array)
        raw_synthetic = generator.generateData(n_required)

        if len(raw_synthetic) != n_required:
            raise ValueError(
                f"XConvGeN returned {len(raw_synthetic)} rows; "
                f"expected {n_required}"
            )

        synthetic = clean_synthetic(raw_synthetic, train)
        validate_output(synthetic, train)
        synthetic.to_csv(output_path, index=False)

        elapsed = time.perf_counter() - start_time
        duplicate_count = int(len(synthetic) - len(synthetic.drop_duplicates()))

        metadata.update(
            {
                "status": "success",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": elapsed,
                "requested_rows": n_required,
                "generated_rows": len(synthetic),
                "column_count": len(synthetic.columns),
                "continuous_columns": continuous_columns,
                "nominal_columns": NOMINAL_COLUMNS,
                "ordinal_columns": ORDINAL_COLUMNS,
                "training_class_counts": train[TARGET]
                .value_counts()
                .sort_index()
                .to_dict(),
                "synthetic_class_counts": synthetic[TARGET]
                .value_counts()
                .sort_index()
                .to_dict(),
                "synthetic_duplicate_count": duplicate_count,
                "synthetic_sha256": sha256_file(output_path),
                "environment": {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "numpy": package_version("numpy"),
                    "pandas": package_version("pandas"),
                    "tensorflow": package_version("tensorflow"),
                    "fdc": package_version("fdc"),
                },
            }
        )
    except Exception as error:
        metadata.update(
            {
                "status": "failed",
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.perf_counter() - start_time,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        if output_path.exists():
            output_path.unlink()
    finally:
        metadata_path.write_text(
            json.dumps(json_safe(metadata), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return metadata


def main() -> int:
    args = parse_args()
    folds = [args.fold] if args.fold is not None else list(range(1, 6))
    summaries = []

    print("=" * 80)
    print("XConvGeN SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("Data root:", args.data_root.resolve())
    print("Output root:", args.output_root.resolve())

    for fold_number in folds:
        print(f"\n{'-' * 80}\nFOLD {fold_number}/5\n{'-' * 80}")
        try:
            metadata = generate_fold(
                fold_number=fold_number,
                data_root=args.data_root,
                output_root=args.output_root,
                base_seed=args.base_seed,
                overwrite=args.overwrite,
            )
        except Exception as error:
            metadata = {
                "fold": fold_number,
                "status": "failed",
                "error_type": type(error).__name__,
                "error_message": str(error),
            }

        summaries.append(metadata)
        print("Status:", metadata["status"].upper())
        if metadata["status"] == "success":
            print("Rows:", metadata["generated_rows"])
            print("Classes:", metadata["synthetic_class_counts"])
            print("Duplicates:", metadata["synthetic_duplicate_count"])
            print("Seconds:", round(metadata["elapsed_seconds"], 2))
        else:
            print("Reason:", metadata.get("error_message", "unknown error"))

    successful = sum(item["status"] == "success" for item in summaries)
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
