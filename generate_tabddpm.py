#!/usr/bin/env python3
"""Train official TabDDPM separately on each fibrosis fold and save 104 rows."""

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
import tempfile
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/tabddpm"))
    parser.add_argument("--tabddpm-root", type=Path, default=Path("external/tab-ddpm"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def configure_imports(root: Path) -> None:
    root = root.resolve()
    scripts = root / "scripts"
    if not root.is_dir() or not scripts.is_dir():
        raise FileNotFoundError(f"Invalid TabDDPM repository: {root}")
    os.environ["PROJECT_DIR"] = str(root)
    os.environ["PYTHONNOUSERSITE"] = "1"
    for path in (str(scripts), str(root)):
        if path not in sys.path:
            sys.path.insert(0, path)


def seed_everything(seed: int) -> None:
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
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def encode_fold(train: pd.DataFrame):
    numerical_columns = [
        column for column in train.columns
        if column not in CATEGORICAL_COLUMNS + [TARGET]
    ]
    category_maps = {}
    encoded_categories = []
    for column in CATEGORICAL_COLUMNS:
        values = sorted(train[column].dropna().unique().tolist())
        mapping = {value: index for index, value in enumerate(values)}
        category_maps[column] = values
        encoded_categories.append(train[column].map(mapping).to_numpy(dtype=np.int64))

    x_num = train[numerical_columns].to_numpy(dtype=np.float32)
    x_cat = np.column_stack(encoded_categories).astype(np.int64)
    y = train[TARGET].map({1: 0, 2: 1, 3: 2}).to_numpy(dtype=np.int64)
    if np.isnan(x_num).any():
        raise ValueError("Numerical training data contain missing values")
    return x_num, x_cat, y, numerical_columns, category_maps


def write_tabddpm_dataset(path: Path, x_num, x_cat, y) -> None:
    path.mkdir(parents=True, exist_ok=True)
    # The official loader requires all three split names. All are technical
    # mirrors of the generator-training fold; benchmark validation is untouched.
    for split in ("train", "val", "test"):
        np.save(path / f"X_num_{split}.npy", x_num)
        np.save(path / f"X_cat_{split}.npy", x_cat)
        np.save(path / f"y_{split}.npy", y)
    info = {
        "name": "fibrosis_fold",
        "id": "fibrosis_fold",
        "task_type": "multiclass",
        "n_classes": 3,
        "n_num_features": int(x_num.shape[1]),
        "n_cat_features": int(x_cat.shape[1]),
        "train_size": int(len(y)),
        "val_size": int(len(y)),
        "test_size": int(len(y)),
    }
    (path / "info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")


def decode_output(artifact_dir, train, numerical_columns, category_maps):
    x_num = np.load(artifact_dir / "X_num_train.npy", allow_pickle=True)
    x_cat = np.load(artifact_dir / "X_cat_train.npy", allow_pickle=True)
    y = np.load(artifact_dir / "y_train.npy", allow_pickle=True)
    x_num = np.asarray(x_num)[:len(train)]
    x_cat = np.asarray(x_cat)[:len(train)]
    y = np.asarray(y).reshape(-1)[:len(train)]

    result = pd.DataFrame(index=range(len(y)), columns=train.columns)
    for index, column in enumerate(numerical_columns):
        result[column] = pd.to_numeric(x_num[:, index], errors="coerce")
    for index, column in enumerate(CATEGORICAL_COLUMNS):
        codes = np.rint(pd.to_numeric(x_cat[:, index], errors="coerce")).astype(int)
        values = category_maps[column]
        if np.any((codes < 0) | (codes >= len(values))):
            raise ValueError(f"Invalid generated category code for {column}")
        result[column] = [values[code] for code in codes]
    y_codes = np.rint(pd.to_numeric(y, errors="coerce")).astype(int)
    if np.any((y_codes < 0) | (y_codes > 2)):
        raise ValueError("Invalid generated diagnosis code")
    result[TARGET] = y_codes + 1
    result = result.loc[:, train.columns]
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def validate(synthetic: pd.DataFrame, train: pd.DataFrame) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic columns do not match train.csv")
    if len(synthetic) != len(train):
        raise ValueError(f"Expected {len(train)} valid rows; obtained {len(synthetic)}")
    classes = sorted(synthetic[TARGET].astype(int).unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Generated classes are {classes}; expected {EXPECTED_CLASSES}")


def generate_fold(fold: int, args, train_fn, sample_fn, device) -> dict:
    fold_name = f"fold_{fold:02d}"
    train_path = args.data_root / "folds" / fold_name / "train.csv"
    output_dir = args.output_root / fold_name
    output_path = output_dir / "synthetic.csv"
    metadata_path = output_dir / "generation_metadata.json"
    artifact_dir = output_dir / "model_artifacts"
    seed = args.base_seed + fold
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    if args.overwrite and artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    record = {
        "model": "TabDDPM", "fold": fold, "seed": seed, "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path), "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "steps": args.steps, "batch_size": args.batch_size,
            "sample_batch_size": args.sample_batch_size,
            "num_timesteps": args.timesteps, "lr": args.lr,
            "weight_decay": args.weight_decay, "scheduler": "cosine",
            "normalization": "quantile", "model": "MLP [256, 256]",
            "numerical_features": 53, "categorical_features": 7,
            "target_conditioning": True,
        },
    }
    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
        x_num, x_cat, y, numerical_columns, category_maps = encode_fold(train)
        with tempfile.TemporaryDirectory(prefix=f"tabddpm_{fold_name}_") as temp:
            data_dir = Path(temp) / "data"
            write_tabddpm_dataset(data_dir, x_num, x_cat, y)
            model_params = {
                "num_classes": 3,
                "is_y_cond": True,
                "rtdl_params": {"d_layers": [256, 256], "dropout": 0.0},
            }
            transform = {
                "seed": seed, "normalization": "quantile",
                "num_nan_policy": None, "cat_nan_policy": None,
                "cat_min_frequency": None, "cat_encoding": None,
                "y_policy": "default",
            }
            train_fn(
                parent_dir=str(artifact_dir), real_data_path=str(data_dir),
                steps=args.steps, lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.batch_size, model_type="mlp",
                model_params=model_params, num_timesteps=args.timesteps,
                gaussian_loss_type="mse", scheduler="cosine", T_dict=transform,
                num_numerical_features=len(numerical_columns), device=device,
                seed=seed, change_val=False,
            )
            sample_fn(
                parent_dir=str(artifact_dir), real_data_path=str(data_dir),
                batch_size=args.sample_batch_size, num_samples=len(train),
                model_type="mlp", model_params=model_params, model_path=str(artifact_dir / "model.pt"),
                num_timesteps=args.timesteps, gaussian_loss_type="mse",
                scheduler="cosine", T_dict=transform,
                num_numerical_features=len(numerical_columns), disbalance=None,
                device=device, seed=seed, change_val=False,
            )
        synthetic = decode_output(artifact_dir, train, numerical_columns, category_maps)
        validate(synthetic, train)
        synthetic.to_csv(output_path, index=False)
        record.update({
            "status": "success", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "requested_rows": len(train), "generated_rows": len(synthetic),
            "training_class_counts": train[TARGET].value_counts().sort_index().to_dict(),
            "synthetic_class_counts": synthetic[TARGET].value_counts().sort_index().to_dict(),
            "synthetic_duplicates": int(synthetic.duplicated().sum()),
            "numerical_columns": numerical_columns, "category_maps": category_maps,
        })
        print("Status: SUCCESS")
        print("Rows:", len(synthetic))
        print("Classes:", synthetic[TARGET].value_counts().sort_index().to_dict())
        print("Duplicates:", int(synthetic.duplicated().sum()))
        print("Seconds:", round(record["elapsed_seconds"], 2))
    except Exception as error:
        record.update({
            "status": "failed", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error_type": type(error).__name__, "error": str(error),
            "traceback": traceback.format_exc(),
        })
        print("Status: FAILED")
        print("Reason:", error)
    record["software"] = {
        "python": sys.version, "platform": platform.platform(),
        "torch": torch.__version__, "numpy": version("numpy"),
        "pandas": version("pandas"), "scikit-learn": version("scikit-learn"),
    }
    metadata_path.write_text(json.dumps(json_safe(record), indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def main() -> None:
    args = parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()
    args.tabddpm_root = args.tabddpm_root.resolve()
    configure_imports(args.tabddpm_root)
    from train import train as train_fn
    from sample import sample as sample_fn

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    folds = [args.fold] if args.fold else list(range(1, 6))
    print("=" * 80)
    print("TabDDPM SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("Device:", device)
    print("Steps:", args.steps)
    results = []
    for fold in folds:
        print("\n" + "-" * 80)
        print(f"FOLD {fold}/5")
        print("-" * 80)
        results.append(generate_fold(fold, args, train_fn, sample_fn, device))
    successes = sum(r["status"] == "success" for r in results)
    print(f"\nSuccessful folds: {successes}/{len(results)}")
    if successes != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
