#!/usr/bin/env python3
"""Generate differentially private DP-CTGAN data for fibrosis folds."""

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
from snsynth import Synthesizer
from snsynth.transform import MinMaxTransformer, TableTransformer


TARGET = "diagnosis"
EXPECTED_CLASSES = [1, 2, 3]
BASE_SEED = 42
NOMINAL_COLUMNS = ["sex", "bariatricSurgery"]
ORDINAL_COLUMNS = [
    "DiabetesMellitusII", "ArterielleHypertonie", "Hyperlipidämie",
    "Hypertriglycerinämie", "KHK",
    # Quantized assay variables with only a small set of valid levels.
    # Declared explicitly as schema information so SmartNoise does not try
    # to infer unstable continuous min/max bounds for them.
    "TrimethylamineNoxide", "Asparagine", "Succinicacid",
    "Acetoaceticacid", TARGET,
]
CATEGORICAL_COLUMNS = NOMINAL_COLUMNS + ORDINAL_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/dpctgan"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--preprocessor-epsilon", type=float, default=0.5)
    parser.add_argument(
        "--bounds-file",
        type=Path,
        default=Path("fibrosis_benchmark_data/metadata/dpctgan_bounds.json"),
    )
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=5.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
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


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def clean(raw, train: pd.DataFrame) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        synthetic = raw.copy()
        if synthetic.columns.tolist() != train.columns.tolist():
            if synthetic.shape[1] != train.shape[1]:
                raise ValueError("DP-CTGAN output has the wrong number of columns")
            synthetic.columns = train.columns
    else:
        synthetic = pd.DataFrame(raw, columns=train.columns)
    synthetic = synthetic.loc[:, train.columns]
    for column in synthetic.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    for column in CATEGORICAL_COLUMNS:
        synthetic[column] = synthetic[column].round().astype(int)
        allowed = set(train[column].astype(int).unique())
        if not set(synthetic[column].unique()).issubset(allowed):
            raise ValueError(f"Invalid generated category in {column}")
    return synthetic


def validate(synthetic: pd.DataFrame, train: pd.DataFrame) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic columns do not match train.csv")
    if len(synthetic) != len(train):
        raise ValueError(f"Expected {len(train)} rows; obtained {len(synthetic)}")
    classes = sorted(synthetic[TARGET].astype(int).unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Generated classes are {classes}; expected {EXPECTED_CLASSES}")


def generate_fold(fold: int, args, use_cuda: bool) -> dict:
    fold_name = f"fold_{fold:02d}"
    train_path = args.data_root.resolve() / "folds" / fold_name / "train.csv"
    output_dir = args.output_root.resolve() / fold_name
    output_path = output_dir / "synthetic.csv"
    metadata_path = output_dir / "generation_metadata.json"
    seed = args.base_seed + fold
    bounds_path = args.bounds_file.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    if args.overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()

    started = time.perf_counter()
    record = {
        "model": "DP-CTGAN", "fold": fold, "seed": seed, "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path), "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "privacy": {
            "definition": "approximate differential privacy conditional on frozen preprocessing bounds",
            "epsilon_total": args.epsilon,
            "epsilon_preprocessing": 0.0,
            "epsilon_model": args.epsilon,
            "delta": args.delta,
            "end_to_end_dp": False,
            "limitation": "bounds were derived from benchmark_full.csv and are not a DP release",
        },
        "configuration": {
            "epochs": args.epochs, "batch_size": args.batch_size,
            "sigma": args.sigma, "max_per_sample_grad_norm": args.max_grad_norm,
            "cuda": use_cuda, "pac": 1,
            "benchmark_validation_used": False,
        },
    }
    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        if len(train) != 104 or train.shape[1] != 61:
            raise ValueError(f"Unexpected training shape: {train.shape}")
        if train.isna().any().any():
            raise ValueError("Training data contain missing values")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
        continuous = [c for c in train.columns if c not in CATEGORICAL_COLUMNS]

        bounds_payload = json.loads(bounds_path.read_text(encoding="utf-8"))
        bounds = bounds_payload["bounds"]
        missing_bounds = sorted(set(continuous) - set(bounds))
        if missing_bounds:
            raise ValueError(f"Missing fixed bounds for: {missing_bounds}")

        constraints = {
            column: MinMaxTransformer(
                lower=float(bounds[column]["lower"]),
                upper=float(bounds[column]["upper"]),
            )
            for column in continuous
        }
        transformer = TableTransformer.create(
            train,
            style="gan",
            nullable=False,
            categorical_columns=NOMINAL_COLUMNS,
            ordinal_columns=ORDINAL_COLUMNS,
            continuous_columns=continuous,
            constraints=constraints,
        )

        model = Synthesizer.create(
            "dpctgan", epsilon=args.epsilon, delta=args.delta,
            epochs=args.epochs, batch_size=args.batch_size,
            sigma=args.sigma, max_per_sample_grad_norm=args.max_grad_norm,
            pac=1, cuda=use_cuda, verbose=True,
        )
        model.fit(
            train,
            transformer=transformer,
            preprocessor_eps=0.0,
            nullable=False,
        )
        synthetic = clean(model.sample(len(train)), train)
        validate(synthetic, train)
        synthetic.to_csv(output_path, index=False)
        record.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "generated_rows": len(synthetic),
            "class_counts": {str(k): int(v) for k, v in synthetic[TARGET].value_counts().sort_index().items()},
            "synthetic_duplicates": int(synthetic.duplicated().sum()),
            "versions": {
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "smartnoise-synth": version("smartnoise-synth"),
                "opacus": version("opacus"),
            },
            "bounds_file": str(bounds_path),
            "bounds_sha256": sha256_file(bounds_path),
        })
        print("Status: SUCCESS")
        print("Rows:", len(synthetic))
        print("Classes:", record["class_counts"])
        print("Duplicates:", record["synthetic_duplicates"])
        return record
    except Exception as error:
        record.update({
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        print("Status: FAILED")
        print("Reason:", error)
        return record
    finally:
        metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.cpu
    print("=" * 80)
    print("DP-CTGAN SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if use_cuda:
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Privacy: epsilon={args.epsilon}, delta={args.delta}")
    folds = [args.fold] if args.fold else list(range(1, 6))
    successful = 0
    for fold in folds:
        print("\n" + "-" * 80)
        print(f"FOLD {fold}/5")
        print("-" * 80)
        result = generate_fold(fold, args, use_cuda)
        successful += result["status"] == "success"
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
