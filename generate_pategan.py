#!/usr/bin/env python3
"""Generate PATEGAN synthetic data for the fibrosis benchmark folds."""

from __future__ import annotations

import argparse
import json
import platform
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from snsynth import Synthesizer

from generate_dpgan import (
    TARGET,
    clean,
    make_transformer,
    package_version,
    seed_everything,
    sha256_file,
    validate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--output-root", type=Path, default=Path("synthetic_data/pategan"))
    parser.add_argument(
        "--bounds-file", type=Path,
        default=Path("fibrosis_benchmark_data/metadata/dpctgan_bounds.json"),
    )
    parser.add_argument("--fold", type=int, choices=range(1, 6), default=None)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=5.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--teacher-iters", type=int, default=5)
    parser.add_argument("--student-iters", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate_fold(fold: int, args) -> dict:
    fold_name = f"fold_{fold:02d}"
    train_path = args.data_root.resolve() / "folds" / fold_name / "train.csv"
    bounds_path = args.bounds_file.resolve()
    output_dir = args.output_root.resolve() / fold_name
    output_path = output_dir / "synthetic.csv"
    metadata_path = output_dir / "generation_metadata.json"
    seed = args.base_seed + fold
    output_dir.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    if args.overwrite:
        for path in (output_path, metadata_path):
            if path.exists():
                path.unlink()

    started = time.perf_counter()
    record = {
        "model": "PATEGAN", "fold": fold, "seed": seed, "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "train_file": str(train_path), "train_sha256": sha256_file(train_path),
        "bounds_file": str(bounds_path), "bounds_sha256": sha256_file(bounds_path),
        "output_file": str(output_path),
        "privacy": {
            "epsilon_target": args.epsilon, "delta": args.delta,
            "epsilon_preprocessing": 0.0,
            "accounting_status": "SmartNoise PATEGAN moments-accountant estimate",
            "end_to_end_dp": False,
            "limitation": "frozen bounds were derived from benchmark_full.csv",
        },
        "configuration": {
            "batch_size": args.batch_size, "latent_dim": args.latent_dim,
            "teacher_iters": args.teacher_iters,
            "student_iters": args.student_iters,
            "binary": False, "benchmark_validation_used": False,
        },
    }

    try:
        seed_everything(seed)
        train = pd.read_csv(train_path)
        if train.shape != (104, 61):
            raise ValueError(f"Unexpected training shape: {train.shape}")
        if train.isna().any().any():
            raise ValueError("Training data contain missing values")
        train[TARGET] = pd.to_numeric(train[TARGET], errors="raise").astype(int)

        transformer = make_transformer(train, bounds_path)
        model = Synthesizer.create(
            "pategan",
            epsilon=args.epsilon,
            delta=args.delta,
            binary=False,
            latent_dim=args.latent_dim,
            batch_size=args.batch_size,
            teacher_iters=args.teacher_iters,
            student_iters=args.student_iters,
        )
        model.fit(train, transformer=transformer, preprocessor_eps=0.0, nullable=False)
        synthetic = clean(model.sample(len(train)), train)
        validate(synthetic, train)
        synthetic.to_csv(output_path, index=False)

        record.update({
            "status": "success",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "generated_rows": len(synthetic),
            "class_counts": {
                str(k): int(v)
                for k, v in synthetic[TARGET].value_counts().sort_index().items()
            },
            "synthetic_duplicates": int(synthetic.duplicated().sum()),
            "versions": {
                "python": platform.python_version(), "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__,
                "smartnoise-synth": package_version("smartnoise-synth"),
            },
        })
        print("Status: SUCCESS")
        print("Rows:", len(synthetic))
        print("Classes:", record["class_counts"])
        print("Duplicates:", record["synthetic_duplicates"])
    except Exception as error:
        record.update({
            "status": "failed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.perf_counter() - started,
            "error": str(error), "traceback": traceback.format_exc(),
        })
        print("Status: FAILED")
        print("Reason:", error)
    finally:
        metadata_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    return record


def main() -> int:
    args = parse_args()
    print("=" * 80)
    print("PATEGAN SEPARATE SYNTHETIC-DATA GENERATION")
    print("=" * 80)
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    print(f"Privacy target: epsilon={args.epsilon}, delta={args.delta}")

    folds = [args.fold] if args.fold else list(range(1, 6))
    results = []
    for fold in folds:
        print("\n" + "-" * 80)
        print(f"FOLD {fold}/5")
        print("-" * 80)
        results.append(generate_fold(fold, args))
    successful = sum(result["status"] == "success" for result in results)
    print(f"\nSuccessful folds: {successful}/{len(folds)}")
    return 0 if successful == len(folds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
