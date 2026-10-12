#!/usr/bin/env python3
"""Generate size-matched Tabula data for the five fibrosis folds."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/tabula"))
    parser.add_argument("--tabula-root", type=Path, default=Path("external/tabula"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=484)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--generation-batch-size", "--top-k",
        dest="generation_batch_size", type=int, default=16,
        help="Rows generated simultaneously (Tabula's argument named k)",
    )
    parser.add_argument("--llm", default="distilgpt2")
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Smoke-test sample count; omit for the required 104-row benchmark output",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_local_import(root: Path) -> None:
    root = root.resolve()
    if not (root / "tabula" / "__init__.py").exists():
        raise FileNotFoundError(f"Invalid Tabula repository: {root}")
    os.environ["PYTHONNOUSERSITE"] = "1"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


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


def safe_training_table(train: pd.DataFrame):
    """Use compact ASCII names to shorten sequences and avoid whitespace."""
    original_to_safe = {column: f"c{index}" for index, column in enumerate(train.columns)}
    safe_to_original = {value: key for key, value in original_to_safe.items()}
    safe = train.rename(columns=original_to_safe).copy()
    categorical = [original_to_safe[column] for column in CATEGORICAL_COLUMNS]
    return safe, categorical, safe_to_original


def clean_sample(raw: pd.DataFrame, train: pd.DataFrame,
                 safe_to_original: dict[str, str]) -> pd.DataFrame:
    if not isinstance(raw, pd.DataFrame):
        raw = pd.DataFrame(raw)
    synthetic = raw.rename(columns=safe_to_original)
    missing = sorted(set(train.columns) - set(synthetic.columns))
    if missing:
        raise ValueError(f"Tabula output is missing columns: {missing}")
    synthetic = synthetic.loc[:, train.columns].copy()
    for column in synthetic.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    for column in CATEGORICAL_COLUMNS:
        synthetic[column] = synthetic[column].round().astype(int)
        allowed = set(train[column].astype(int).unique())
        synthetic = synthetic[synthetic[column].isin(allowed)]
    return synthetic.reset_index(drop=True)


def validate(synthetic: pd.DataFrame, train: pd.DataFrame,
             expected_rows: int, require_all_classes: bool) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic column order does not match train.csv")
    if len(synthetic) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} valid rows; obtained {len(synthetic)}"
        )
    if require_all_classes:
        classes = sorted(synthetic[TARGET].astype(int).unique().tolist())
        if classes != EXPECTED_CLASSES:
            raise ValueError(f"Generated diagnosis classes are {classes}; expected {EXPECTED_CLASSES}")


def generate_fold(fold: int, args, Tabula) -> dict:
    fold_name = f"fold_{fold:02d}"
    train_path = args.data_root.resolve() / "folds" / fold_name / "train.csv"
    output_dir = args.output_root.resolve() / fold_name
    expected_rows = args.n_samples if args.n_samples is not None else 104
    is_smoke = expected_rows != 104
    output_path = output_dir / (f"synthetic_smoke_{expected_rows}.csv" if is_smoke else "synthetic.csv")
    metadata_path = output_dir / ("generation_metadata_smoke.json" if is_smoke else "generation_metadata.json")
    artifact_dir = output_dir / "model_artifacts"
    experiment_dir = artifact_dir / "trainer"
    saved_model_dir = artifact_dir / "saved_model"
    seed = args.base_seed + fold

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    if args.overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    record = {
        "model": "Tabula",
        "fold": fold,
        "seed": seed,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "llm": args.llm,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "temperature": args.temperature,
            "generation_batch_size": args.generation_batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "temporary_column_names": "c0...c60",
            "benchmark_validation_used": False,
            "requested_rows": expected_rows,
            "smoke_test": is_smoke,
        },
    }

    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        if len(train) != 104:
            raise ValueError(f"Expected 104 training rows; found {len(train)}")
        if train.isna().any().any():
            raise ValueError("Training data contain missing values")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
        safe_train, safe_categorical, safe_to_original = safe_training_table(train)

        model = Tabula(
            llm=args.llm,
            experiment_dir=str(experiment_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            categorical_columns=safe_categorical,
        )
        model.fit(safe_train)

        model_save_warning = None
        try:
            model.save(str(saved_model_dir))
        except (TypeError, ValueError) as error:
            # The official save method attempts to JSON-serialize every
            # attribute. Newer compatible Transformers releases expose a
            # GPT2Config object that is not directly JSON serializable.
            # Preserve the actual weights and sufficient reconstruction
            # information without interrupting generation.
            model_save_warning = str(error)
            if saved_model_dir.exists():
                shutil.rmtree(saved_model_dir)
            saved_model_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.model.state_dict(), saved_model_dir / "model.pt")
            reconstruction = {
                "llm": args.llm,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "categorical_columns": safe_categorical,
                "original_columns": train.columns.tolist(),
                "safe_to_original": safe_to_original,
                "note": (
                    "Instantiate Tabula with these parameters, then call "
                    "load_finetuned_model('model.pt')."
                ),
            }
            (saved_model_dir / "reconstruction_config.json").write_text(
                json.dumps(reconstruction, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Tabula's own sample loop continues until it has n_samples complete,
        # numerically valid records, so no extra generation budget is needed.
        requested = expected_rows
        raw = model.sample(
            n_samples=requested,
            temperature=args.temperature,
            k=args.generation_batch_size,
            max_length=args.max_length,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        synthetic = clean_sample(raw, train, safe_to_original)
        if len(synthetic) < expected_rows:
            raise ValueError(
                f"Only {len(synthetic)}/{expected_rows} valid rows remained from {requested} samples"
            )
        synthetic = synthetic.iloc[:expected_rows].reset_index(drop=True)
        validate(synthetic, train, expected_rows, require_all_classes=not is_smoke)
        synthetic.to_csv(output_path, index=False)

        record.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "raw_generated_rows": len(raw),
            "generated_rows": len(synthetic),
            "class_counts": {str(k): int(v) for k, v in synthetic[TARGET].value_counts().sort_index().items()},
            "synthetic_duplicates": int(synthetic.duplicated().sum()),
            "model_directory": str(saved_model_dir),
            "native_model_save_warning": model_save_warning,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": package_version("transformers"),
                "datasets": package_version("datasets"),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
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
            "error": str(error),
            "traceback": traceback.format_exc(),
        })
        print("Status: FAILED")
        print("Reason:", error)
        return record
    finally:
        metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    configure_local_import(args.tabula_root)
    from tabula import Tabula

    print("=" * 80)
    print("Tabula SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    folds = [args.fold] if args.fold else list(range(1, 6))
    successful = 0
    for fold in folds:
        print("\n" + "-" * 80)
        print(f"FOLD {fold}/5")
        print("-" * 80)
        record = generate_fold(fold, args, Tabula)
        successful += record["status"] == "success"
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
