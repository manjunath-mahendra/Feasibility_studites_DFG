#!/usr/bin/env python3
"""Generate size-matched TabDiff data for the fibrosis benchmark folds.

Only ``fibrosis_benchmark_data/folds/fold_XX/train.csv`` is used for model
training.  A one-row internal test file copied from that same training fold is
provided because TabDiff's custom-data preprocessor otherwise removes 10% of
the 104 training rows.  The benchmark validation folds are never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import shutil
import subprocess
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/tabdiff"))
    parser.add_argument("--tabdiff-root", type=Path, default=Path("external/tabdiff"))
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-work-data", action="store_true")
    parser.add_argument("--debug", action="store_true", help="Use TabDiff's short debug training mode")
    return parser.parse_args()


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


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run(command: list[str], cwd: Path, log_path: Path) -> None:
    print("Command:", " ".join(command), flush=True)
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return_code = process.wait()
    if return_code:
        raise RuntimeError(f"Command failed with exit code {return_code}: {' '.join(command)}")


def latest_file(roots: list[Path], filename: str) -> Path:
    matches: list[Path] = []
    for root in roots:
        if root.exists():
            matches.extend(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under: {roots}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def clean_output(sample_path: Path, train: pd.DataFrame) -> pd.DataFrame:
    synthetic = pd.read_csv(sample_path)

    # Some TabDiff outputs use integer column labels. Restore original names
    # by position when the dimensions agree.
    if synthetic.columns.tolist() != train.columns.tolist():
        if synthetic.shape[1] != train.shape[1]:
            raise ValueError(
                f"Generated data have {synthetic.shape[1]} columns; expected {train.shape[1]}"
            )
        synthetic.columns = train.columns

    synthetic = synthetic.loc[:, train.columns].copy()
    for column in synthetic.columns:
        synthetic[column] = pd.to_numeric(synthetic[column], errors="coerce")
    synthetic = synthetic.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

    for column in CATEGORICAL_COLUMNS + [TARGET]:
        synthetic[column] = synthetic[column].round().astype(int)

    # Enforce only valid categories learned from this fold. TabDiff normally
    # already guarantees this; this check detects decoding/configuration bugs.
    for column in CATEGORICAL_COLUMNS + [TARGET]:
        allowed = set(train[column].astype(int).unique())
        observed = set(synthetic[column].unique())
        if not observed.issubset(allowed):
            raise ValueError(f"Invalid generated values for {column}: {sorted(observed - allowed)}")
    return synthetic


def validate(synthetic: pd.DataFrame, train: pd.DataFrame) -> None:
    if synthetic.columns.tolist() != train.columns.tolist():
        raise ValueError("Synthetic column order does not match train.csv")
    if len(synthetic) != len(train):
        raise ValueError(f"Expected {len(train)} valid rows; obtained {len(synthetic)}")
    classes = sorted(synthetic[TARGET].astype(int).unique().tolist())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Generated diagnosis classes are {classes}; expected {EXPECTED_CLASSES}")


def write_metadata(path: Path, dataname: str, train: pd.DataFrame,
                   raw_path: Path, internal_test_path: Path, root: Path) -> None:
    numerical = [column for column in train.columns if column not in CATEGORICAL_COLUMNS + [TARGET]]
    metadata = {
        "name": dataname,
        "task_type": "multiclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": [train.columns.get_loc(column) for column in numerical],
        "cat_col_idx": [train.columns.get_loc(column) for column in CATEGORICAL_COLUMNS],
        "target_col_idx": [train.columns.get_loc(TARGET)],
        "file_type": "csv",
        "data_path": str(raw_path.relative_to(root)),
        "test_path": str(internal_test_path.relative_to(root)),
        "val_path": None,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def generate_fold(fold: int, args: argparse.Namespace) -> dict:
    root = args.tabdiff_root.resolve()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    fold_name = f"fold_{fold:02d}"
    train_path = data_root / "folds" / fold_name / "train.csv"
    output_dir = output_root / fold_name
    output_path = output_dir / "synthetic.csv"
    record_path = output_dir / "generation_metadata.json"
    log_path = output_dir / "tabdiff.log"
    seed = args.base_seed + fold
    dataname = f"fibrosis_{fold_name}"
    exp_name = f"benchmark_seed_{seed}"
    work_data_dir = root / "data" / dataname
    info_path = root / "data" / "Info" / f"{dataname}.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    if args.overwrite:
        for path in (output_path, record_path, log_path):
            if path.exists():
                path.unlink()
        if work_data_dir.exists():
            shutil.rmtree(work_data_dir)
        if info_path.exists():
            info_path.unlink()

    if not train_path.exists():
        raise FileNotFoundError(train_path)
    if not (root / "main.py").exists() or not (root / "process_dataset.py").exists():
        raise FileNotFoundError(f"Invalid TabDiff repository: {root}")

    started = time.perf_counter()
    record = {
        "model": "TabDiff",
        "fold": fold,
        "seed": seed,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path),
        "train_sha256": sha256_file(train_path),
        "output_file": str(output_path),
        "configuration": {
            "gpu": args.gpu,
            "debug": args.debug,
            "deterministic": True,
            "requested_rows": 104,
            "internal_test_source": "first row of current training fold",
            "benchmark_validation_used": False,
        },
    }

    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        if len(train) != 104:
            raise ValueError(f"Expected 104 training rows; found {len(train)}")
        if train.isna().any().any():
            raise ValueError("TabDiff input contains missing values")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)
        missing = sorted(set(CATEGORICAL_COLUMNS + [TARGET]) - set(train.columns))
        if missing:
            raise ValueError(f"Required columns are missing: {missing}")

        work_data_dir.mkdir(parents=True, exist_ok=False)
        info_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path = work_data_dir / f"{dataname}.csv"
        internal_test_path = work_data_dir / "internal_test.csv"
        train.to_csv(raw_path, index=False)
        train.iloc[[0]].to_csv(internal_test_path, index=False)
        write_metadata(info_path, dataname, train, raw_path, internal_test_path, root)

        python = sys.executable
        run([python, "-s", "process_dataset.py", "--dataname", dataname], root, log_path)

        processed_train = pd.read_csv(work_data_dir / "train.csv")
        if len(processed_train) != len(train):
            raise RuntimeError(
                f"TabDiff preprocessing retained {len(processed_train)}/{len(train)} rows"
            )

        common = [
            "--dataname", dataname,
            "--gpu", str(args.gpu),
            "--no_wandb",
            "--deterministic",
            "--exp_name", exp_name,
        ]
        train_command = [python, "-s", "main.py", "--mode", "train", *common]
        if args.debug:
            train_command.append("--debug")
        run(train_command, root, log_path)

        # Reduced runs (<4001 steps) do not create best_ema_model_*.pt, which
        # is the only filename that TabDiff test mode discovers automatically.
        # Select the latest regular checkpoint explicitly instead.
        checkpoint_dir = root / "tabdiff" / "ckpt" / dataname / exp_name
        regular_checkpoints = list(checkpoint_dir.glob("model_*.pt"))
        if not regular_checkpoints:
            raise FileNotFoundError(
                f"No model_*.pt checkpoint found under {checkpoint_dir}. "
                "Set check_val_every no larger than the configured steps."
            )
        checkpoint = max(
            regular_checkpoints,
            key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
        )

        search_roots = [
            root / "tabdiff" / "result" / dataname / exp_name,
            root / "tabdiff" / "result" / dataname,
            root / "synthetic" / dataname,
        ]

        # Test mode writes samples.csv before running TabDiff's optional MLE
        # benchmark.  Our one-row technical test file intentionally contains
        # no held-out benchmark records, so that downstream MLE can fail even
        # though generation succeeded.  Preserve the generated samples in
        # that case; the common external evaluator will assess all generators.
        test_error = None
        try:
            run([
                python, "-s", "main.py", "--mode", "test", *common,
                "--num_samples_to_generate", str(len(train)),
                "--ckpt_path", str(checkpoint),
            ], root, log_path)
        except RuntimeError as error:
            test_error = str(error)
            print(
                "TabDiff post-generation evaluation exited non-zero; "
                "checking whether samples.csv was created.",
                flush=True,
            )

        sample_path = latest_file(search_roots, "samples.csv")
        synthetic = clean_output(sample_path, train)
        validate(synthetic, train)
        synthetic.to_csv(output_path, index=False)

        record.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "generated_rows": len(synthetic),
            "class_counts": {str(k): int(v) for k, v in synthetic[TARGET].value_counts().sort_index().items()},
            "synthetic_duplicates": int(synthetic.duplicated().sum()),
            "tabdiff_sample_file": str(sample_path),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "tabdiff_post_generation_evaluation_error": test_error,
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": package_version("scikit-learn"),
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
        record_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if not args.keep_work_data:
            if work_data_dir.exists():
                shutil.rmtree(work_data_dir)
            if info_path.exists():
                info_path.unlink()


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("TabDiff SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("Python:", sys.executable)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(args.gpu))
    folds = [args.fold] if args.fold else list(range(1, 6))
    successful = 0
    for fold in folds:
        print("\n" + "-" * 80)
        print(f"FOLD {fold}/5")
        print("-" * 80)
        record = generate_fold(fold, args)
        successful += record["status"] == "success"
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
