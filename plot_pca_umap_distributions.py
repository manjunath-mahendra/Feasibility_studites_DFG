#!/usr/bin/env python3
"""Separate real/synthetic PCA and UMAP plots in shared real-fitted spaces."""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import umap
except ImportError as exc:
    raise SystemExit("Install UMAP: python -m pip install umap-learn==0.5.7") from exc


MODELS = ["xconvgen", "ctgan", "ctabgan", "cart", "tabddpm", "tabdiff", "dpctgan", "dpgan"]
NAMES = {
    "xconvgen": "NextConvGeN", "ctgan": "CTGAN", "ctabgan": "CTAB-GAN",
    "cart": "CART", "tabddpm": "TabDDPM", "tabdiff": "TabDiff",
    "dpctgan": "DP-CTGAN (ε=5)", "dpgan": "DPGAN (ε=5)",
}

# Fixed class colors used for real and synthetic data in every fold and model.
CLASS_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
CLASS_NAMES = {1: "Low", 2: "Significant", 3: "High"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data"))
    parser.add_argument("--output-root", type=Path, default=Path("distribution_visualization"))
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--target", default="diagnosis")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def clean_output(path: Path):
    resolved = path.resolve()
    forbidden = {Path("/").resolve(), Path.cwd().resolve(), Path.home().resolve()}
    if resolved in forbidden or resolved.name in {"", ".", ".."}:
        raise ValueError(f"Refusing to delete unsafe path: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
        print(f"Deleted previous results: {resolved}")


def features_and_target(frame: pd.DataFrame, target: str):
    if target not in frame.columns:
        raise ValueError(f"Missing target column {target!r}")
    features = frame.drop(columns=[target]).apply(pd.to_numeric, errors="coerce")
    labels = pd.to_numeric(frame[target], errors="raise").to_numpy(dtype=int)
    return features, labels


def peacock_style_test(sample1: np.ndarray, sample2: np.ndarray):
    """Approximate Peacock-style calculation from the supplied reference code."""
    n1, d1 = sample1.shape
    n2, d2 = sample2.shape
    if d1 != d2:
        raise ValueError("Samples must have equal dimensionality")
    combined = np.vstack([sample1, sample2])
    grid = np.linspace(combined.min(axis=0), combined.max(axis=0), len(combined), axis=0)
    ecdf1 = np.array([np.mean(np.all(sample1 <= point, axis=1)) for point in grid])
    ecdf2 = np.array([np.mean(np.all(sample2 <= point, axis=1)) for point in grid])
    statistic = float(np.max(np.abs(ecdf1 - ecdf2)))
    z_value = statistic * np.sqrt((n1 * n2) / (n1 + n2))
    p_value = float(np.clip(2 * (1 - norm.cdf(z_value)), 0, 1))
    return statistic, p_value


def equal_limits(arrays: list[np.ndarray], padding=0.08):
    points = np.vstack(arrays)
    points = points[np.isfinite(points).all(axis=1)]
    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    span = max(x_max - x_min, y_max - y_min, 1.0) * (1 + padding)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    return (
        (x_center - span / 2, x_center + span / 2),
        (y_center - span / 2, y_center + span / 2),
    )


def plot_dataset(
    xy,
    labels,
    xlabel,
    ylabel,
    limits,
    output_path,
    dpi,
    show_legend=False,
):
    # Use proposal-scale typography for PCA while retaining the existing UMAP
    # styling. PCA font sizes match those used in the utility plots.
    is_pca = xlabel == "PC1" and ylabel == "PC2"
    figure_size = (8, 8) if is_pca else (6.4, 6.4)
    point_size = 110 if is_pca else 44
    axis_font_size = 30 if is_pca else 15
    tick_font_size = 25 if is_pca else 12
    legend_font_size = 22 if is_pca else 11
    legend_title_size = 24 if is_pca else 12

    fig, ax = plt.subplots(figsize=figure_size)
    for diagnosis in sorted(np.unique(labels)):
        mask = labels == diagnosis
        ax.scatter(
            xy[mask, 0], xy[mask, 1],
            color=CLASS_COLORS.get(int(diagnosis), "#7f7f7f"),
            marker="o", s=point_size, alpha=0.75,
            edgecolors="white", linewidths=0.7 if is_pca else 0.45,
            label=f"{int(diagnosis)}: {CLASS_NAMES.get(int(diagnosis), diagnosis)}",
        )
    ax.set_xlim(*limits[0]); ax.set_ylim(*limits[1])
    ax.set_aspect("equal", adjustable="box"); ax.set_box_aspect(1)
    ax.set_xlabel(xlabel, fontsize=axis_font_size)
    ax.set_ylabel(ylabel, fontsize=axis_font_size)
    ax.tick_params(axis="both", labelsize=tick_font_size)
    if show_legend:
        ax.legend(
            title="Diagnosis",
            fontsize=legend_font_size,
            title_fontsize=legend_title_size,
            loc="best",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(output_path, dpi=dpi, bbox_inches="tight"); plt.close(fig)


def add_coordinates(rows, method, fold, model, source, labels, xy):
    for row_index, (label, point) in enumerate(zip(labels, xy)):
        rows.append({
            "Method": method, "Fold": fold, "Model": model, "Source": source,
            "Row": row_index, "Diagnosis": int(label),
            "Component_1": float(point[0]), "Component_2": float(point[1]),
        })


def main():
    args = parse_args()
    if args.clean_output:
        clean_output(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    coordinate_rows, test_rows, fold_metadata = [], [], {}

    for fold in args.folds:
        fold_name = f"fold_{fold:02d}"
        real_path = args.data_root / "folds" / fold_name / "train.csv"
        if not real_path.exists():
            warnings.warn(f"Missing real data: {real_path}"); continue

        real = pd.read_csv(real_path)
        real_x, real_y = features_and_target(real, args.target)

        # Fit preprocessing, PCA, and UMAP exactly once on real training data.
        preprocessor = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        real_scaled = preprocessor.fit_transform(real_x)

        pca = PCA(n_components=2)
        real_pca = pca.fit_transform(real_scaled)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=min(args.umap_neighbors, max(2, len(real) - 1)),
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=args.seed + fold,
            transform_seed=args.seed + fold,
            n_jobs=1,
        )
        real_umap = reducer.fit_transform(real_scaled)

        projections = {
            "real": {"labels": real_y, "pca": real_pca, "umap": real_umap},
            "models": {},
        }
        print(f"FOLD {fold}: real={len(real)}")

        for model in args.models:
            synthetic_path = args.synthetic_root / model / fold_name / "synthetic.csv"
            if not synthetic_path.exists():
                warnings.warn(f"Missing: {synthetic_path}"); continue
            synthetic = pd.read_csv(synthetic_path)
            missing = [column for column in real.columns if column not in synthetic.columns]
            extra = [column for column in synthetic.columns if column not in real.columns]
            if missing or extra:
                warnings.warn(f"Skipping {synthetic_path}: missing={missing}, extra={extra}"); continue
            synthetic = synthetic.loc[:, real.columns]
            try:
                synthetic_x, synthetic_y = features_and_target(synthetic, args.target)
                synthetic_scaled = preprocessor.transform(synthetic_x)
                synthetic_pca = pca.transform(synthetic_scaled)
                synthetic_umap = reducer.transform(synthetic_scaled)
                statistic, p_value = peacock_style_test(real_pca, synthetic_pca)
                projections["models"][model] = {
                    "labels": synthetic_y, "pca": synthetic_pca, "umap": synthetic_umap,
                }
                test_rows.append({
                    "Fold": fold, "Model": NAMES.get(model, model),
                    "Peacock_Style_Statistic": statistic,
                    "Peacock_Style_P_Value": p_value,
                    "Test_Space": "Shared real-fitted PCA",
                })
                print(f"  {NAMES.get(model, model)}={len(synthetic)}, p={p_value:.5g}")
            except Exception as exc:
                warnings.warn(f"Skipping {synthetic_path}: {exc}")

        if not projections["models"]:
            continue

        # Identical square limits for real and all model plots within this fold.
        pca_limits = equal_limits([real_pca] + [x["pca"] for x in projections["models"].values()])
        umap_limits = equal_limits([real_umap] + [x["umap"] for x in projections["models"].values()])

        plot_dataset(
            real_pca, real_y, "PC1", "PC2",
            pca_limits, args.output_root / "pca" / fold_name / "real.png",
            args.dpi, show_legend=True,
        )
        plot_dataset(
            real_umap, real_y, "UMAP1", "UMAP2", umap_limits,
            args.output_root / "umap" / fold_name / "real.png",
            args.dpi, show_legend=True,
        )
        add_coordinates(coordinate_rows, "PCA", fold, "Real", "Real", real_y, real_pca)
        add_coordinates(coordinate_rows, "UMAP", fold, "Real", "Real", real_y, real_umap)

        for model, values in projections["models"].items():
            display = NAMES.get(model, model)
            plot_dataset(
                values["pca"], values["labels"], "PC1", "PC2", pca_limits,
                args.output_root / "pca" / fold_name / f"{model}.png",
                args.dpi, show_legend=False,
            )
            plot_dataset(
                values["umap"], values["labels"], "UMAP1", "UMAP2", umap_limits,
                args.output_root / "umap" / fold_name / f"{model}.png",
                args.dpi, show_legend=False,
            )
            add_coordinates(coordinate_rows, "PCA", fold, display, "Synthetic", values["labels"], values["pca"])
            add_coordinates(coordinate_rows, "UMAP", fold, display, "Synthetic", values["labels"], values["umap"])

        fold_metadata[fold_name] = {
            "real_rows": len(real),
            "models": list(projections["models"]),
            "pca_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "pca_limits": pca_limits, "umap_limits": umap_limits,
        }

    pd.DataFrame(coordinate_rows).to_csv(args.output_root / "projection_coordinates.csv", index=False)
    pd.DataFrame(test_rows).to_csv(args.output_root / "peacock_style_test_results.csv", index=False)
    metadata = {
        "methodology": "Per fold, preprocessing, PCA and UMAP were fitted on real training data only. Each synthetic dataset was transformed using those unchanged fitted objects. Plots are separate.",
        "target_usage": "diagnosis excluded from fitting and used only to color points",
        "class_colors": {str(k): v for k, v in CLASS_COLORS.items()},
        "marker": "circle for all points",
        "plot_makeup": "No titles or grids. Legends are shown only on real-data plots.",
        "axis_policy": "equal x/y span and identical limits across real and all models within each fold and method",
        "folds": fold_metadata,
    }
    with open(args.output_root / "visualization_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, ensure_ascii=False)
    print(f"Saved: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
