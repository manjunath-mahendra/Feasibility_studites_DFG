#!/usr/bin/env python3
"""Create frozen, data-derived benchmark bounds for DP-CTGAN.

These bounds are suitable for a feasibility benchmark, but are not public
domain bounds. Therefore, epsilon applies to DP-CTGAN training conditional on
this frozen preprocessing schema, not to the process that created the bounds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


TARGET = "diagnosis"
CATEGORICAL_COLUMNS = [
    "sex", "bariatricSurgery", "DiabetesMellitusII",
    "ArterielleHypertonie", "Hyperlipidämie",
    "Hypertriglycerinämie", "KHK", TARGET,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("fibrosis_benchmark_data/benchmark_full.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("fibrosis_benchmark_data/metadata/dpctgan_bounds.json"),
    )
    parser.add_argument("--margin", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; use --overwrite")
    if args.margin < 0:
        raise ValueError("--margin must be non-negative")

    data = pd.read_csv(input_path)
    continuous = [c for c in data.columns if c not in CATEGORICAL_COLUMNS]
    bounds = {}

    for column in continuous:
        values = pd.to_numeric(data[column], errors="raise").to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError(f"{column!r} contains no finite values")

        observed_min = float(values.min())
        observed_max = float(values.max())
        span = observed_max - observed_min

        if span == 0:
            scale = max(abs(observed_min), 1.0)
            lower = observed_min - args.margin * scale
            upper = observed_max + args.margin * scale
        else:
            lower = observed_min - args.margin * span
            upper = observed_max + args.margin * span

        # All-positive laboratory quantities retain a meaningful zero floor.
        if observed_min >= 0:
            lower = 0.0

        if not (math.isfinite(lower) and math.isfinite(upper) and lower < upper):
            raise ValueError(f"Invalid bounds for {column!r}: {lower}, {upper}")

        bounds[column] = {
            "lower": float(lower),
            "upper": float(upper),
            "observed_min": observed_min,
            "observed_max": observed_max,
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": str(input_path),
        "source_sha256": sha256_file(input_path),
        "method": "full-dataset observed range expanded by a fixed margin",
        "margin_fraction": args.margin,
        "privacy_status": "data-derived benchmark metadata; not a DP release",
        "dp_interpretation": (
            "epsilon applies to DP-CTGAN training conditional on these frozen "
            "bounds; it does not cover creation of this bounds file"
        ),
        "categorical_columns": CATEGORICAL_COLUMNS,
        "continuous_columns": continuous,
        "bounds": bounds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("Saved:", output_path)
    print("Continuous columns:", len(continuous))
    print("Margin:", args.margin)
    print("WARNING:", payload["privacy_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
