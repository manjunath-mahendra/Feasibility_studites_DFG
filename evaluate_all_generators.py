#!/usr/bin/env python3
"""Common utility, fidelity, and privacy benchmark for saved synthetic data."""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize


TARGET = "diagnosis"
CLASSES = [1, 2, 3]
DEFAULT_MODELS = [
    "xconvgen", "ctgan", "ctabgan", "cart",
    "tabddpm", "tabdiff", "dpctgan", "dpgan",
]
DISPLAY_NAMES = {
    "xconvgen": "NextConvGeN", "ctgan": "CTGAN", "ctabgan": "CTAB-GAN",
    "cart": "CART", "tabddpm": "TabDDPM", "tabdiff": "TabDiff",
    "dpctgan": "DP-CTGAN (ε=5)", "dpgan": "DPGAN (ε=5)",
}
NOMINAL = ["sex", "bariatricSurgery"]
ORDINAL = [
    "DiabetesMellitusII", "ArterielleHypertonie", "Hyperlipidämie",
    "Hypertriglycerinämie", "KHK",
]
CATEGORICAL = NOMINAL + ORDINAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data"))
    parser.add_argument("--output-root", type=Path, default=Path("benchmark_results"))
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return df[columns].apply(pd.to_numeric, errors="coerce")


def classifier_pipeline(train: pd.DataFrame, trees: int, seed: int) -> Pipeline:
    features = [c for c in train.columns if c != TARGET]
    categorical = [c for c in CATEGORICAL if c in features]
    continuous = [c for c in features if c not in categorical]
    preprocessor = ColumnTransformer([
        ("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), continuous),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=trees, class_weight="balanced", random_state=seed,
            n_jobs=-1, max_features="sqrt",
        )),
    ])


def macro_auc(y_true, probabilities, fitted_classes) -> float:
    aligned = np.zeros((len(y_true), len(CLASSES)), dtype=float)
    for source_index, label in enumerate(fitted_classes):
        if int(label) in CLASSES:
            aligned[:, CLASSES.index(int(label))] = probabilities[:, source_index]
    y_binary = label_binarize(y_true, classes=CLASSES)
    return float(roc_auc_score(y_binary, aligned, average="macro", multi_class="ovr"))


def evaluate_utility(train: pd.DataFrame, validation: pd.DataFrame, trees: int, seed: int) -> dict:
    x_train = train.drop(columns=[TARGET])
    y_train = pd.to_numeric(train[TARGET], errors="raise").astype(int)
    x_test = validation.drop(columns=[TARGET])
    y_test = pd.to_numeric(validation[TARGET], errors="raise").astype(int)
    model = classifier_pipeline(train, trees, seed)
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    probability = model.predict_proba(x_test)
    fitted_classes = model.named_steps["classifier"].classes_
    return {
        "Accuracy": accuracy_score(y_test, prediction),
        "Balanced Accuracy": balanced_accuracy_score(y_test, prediction),
        "Macro F1": f1_score(y_test, prediction, average="macro", zero_division=0),
        "Weighted F1": f1_score(y_test, prediction, average="weighted", zero_division=0),
        "Macro ROC-AUC": macro_auc(y_test, probability, fitted_classes),
    }


def total_variation(real: pd.Series, synthetic: pd.Series) -> float:
    levels = sorted(set(real.dropna().tolist()) | set(synthetic.dropna().tolist()), key=str)
    p = real.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    q = synthetic.value_counts(normalize=True).reindex(levels, fill_value=0).to_numpy(float)
    return float(0.5 * np.abs(p - q).sum())


def fidelity_metrics(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    categorical = [c for c in CATEGORICAL + [TARGET] if c in real.columns]
    continuous = [c for c in real.columns if c not in categorical]
    ks_values = [
        ks_2samp(pd.to_numeric(real[c]), pd.to_numeric(synthetic[c])).statistic
        for c in continuous
    ]
    tv_values = [total_variation(real[c], synthetic[c]) for c in categorical]

    correlation_columns = continuous + [c for c in ORDINAL + [TARGET] if c in real.columns]
    real_corr = numeric_frame(real, correlation_columns).corr(method="spearman")
    syn_corr = numeric_frame(synthetic, correlation_columns).corr(method="spearman")
    mask = np.triu(np.ones(real_corr.shape, dtype=bool), k=1)
    r = real_corr.to_numpy()[mask]
    s = syn_corr.to_numpy()[mask]
    valid = np.isfinite(r) & np.isfinite(s)
    r, s = r[valid], s[valid]
    structure = spearmanr(r, s).statistic if len(r) > 1 else np.nan
    return {
        "Mean Continuous KS": float(np.mean(ks_values)),
        "Mean Categorical TVD": float(np.mean(tv_values)),
        "Mean Correlation Difference": float(np.mean(np.abs(r - s))) if len(r) else np.nan,
        "Correlation Structure rho": float(structure) if np.isfinite(structure) else np.nan,
        "Correlation Pairs": int(len(r)),
    }


def gower_encode(reference: pd.DataFrame, frames: list[pd.DataFrame]):
    categorical = [c for c in CATEGORICAL + [TARGET] if c in reference.columns]
    continuous = [c for c in reference.columns if c not in categorical]
    ref_num = numeric_frame(reference, continuous)
    lower = ref_num.min(axis=0).to_numpy(float)
    span = (ref_num.max(axis=0) - ref_num.min(axis=0)).to_numpy(float)
    span[span == 0] = 1.0
    encoded = []
    for frame in frames:
        num = numeric_frame(frame, continuous).to_numpy(float)
        num = (num - lower) / span
        cat = frame[categorical].astype(str).to_numpy()
        encoded.append((num, cat))
    return encoded, len(continuous) + len(categorical)


def gower_matrix(a, b, n_features: int) -> np.ndarray:
    a_num, a_cat = a
    b_num, b_cat = b
    numeric_distance = np.abs(a_num[:, None, :] - b_num[None, :, :]).sum(axis=2)
    categorical_distance = (a_cat[:, None, :] != b_cat[None, :, :]).sum(axis=2)
    return (numeric_distance + categorical_distance) / n_features


def privacy_metrics(train: pd.DataFrame, validation: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    encoded, n_features = gower_encode(train, [train, validation, synthetic])
    train_e, validation_e, synthetic_e = encoded
    syn_real = gower_matrix(synthetic_e, train_e, n_features)
    real_real = gower_matrix(train_e, train_e, n_features)
    np.fill_diagonal(real_real, np.inf)
    syn_dcr = syn_real.min(axis=1)
    real_dcr = real_real.min(axis=1)

    member_distance = gower_matrix(train_e, synthetic_e, n_features).min(axis=1)
    nonmember_distance = gower_matrix(validation_e, synthetic_e, n_features).min(axis=1)
    labels = np.r_[np.ones(len(member_distance)), np.zeros(len(nonmember_distance))]
    scores = -np.r_[member_distance, nonmember_distance]

    exact = len(synthetic.merge(train.drop_duplicates(), how="inner", on=list(train.columns)))
    mean_real = float(np.mean(real_dcr))
    return {
        "Exact Real-Synthetic Matches": int(exact),
        "Synthetic Duplicates": int(synthetic.duplicated().sum()),
        "Mean DCR": float(np.mean(syn_dcr)),
        "Median DCR": float(np.median(syn_dcr)),
        "5th Percentile DCR": float(np.percentile(syn_dcr, 5)),
        "Mean Real-Real Distance": mean_real,
        "DCR Ratio": float(np.mean(syn_dcr) / mean_real) if mean_real > 0 else np.nan,
        "MIA AUC": float(roc_auc_score(labels, scores)),
        "Mean Member Distance": float(np.mean(member_distance)),
        "Mean Non-member Distance": float(np.mean(nonmember_distance)),
    }


def mean_sd_table(frame: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    return frame.groupby("Model")[metrics].agg(["mean", "std", "count"]).reset_index()


def plot_label(label: str) -> str:
    """Short publication label used only in figures."""
    return {
        "DP-CTGAN (ε=5)": "DP-CTGAN",
        "DPGAN (ε=5)": "DPGAN",
    }.get(label, label)


def save_bar(
    summary: pd.DataFrame,
    metric: str,
    ylabel: str,
    path: Path,
    reference=None,
    *,
    real_reference: bool = False,
):
    means = summary[(metric, "mean")].to_numpy(float)
    stds = summary[(metric, "std")].fillna(0).to_numpy(float)
    labels = [plot_label(label) for label in summary[("Model", "")].tolist()]

    if real_reference:
        real_index = next(
            (index for index, label in enumerate(labels) if label == "Real (TRTR)"),
            None,
        )
        if real_index is None:
            raise ValueError("Real (TRTR) row is required for the utility reference plot")

        order = [real_index] + [index for index in range(len(labels)) if index != real_index]
        means, stds = means[order], stds[order]
        labels = [labels[index] for index in order]
        # Highlight the real-data reference in orange and use blue for every
        # synthetic generator.
        colors = ["#F28E2B"] + ["#4C78A8"] * (len(labels) - 1)
    else:
        colors = "#4C78A8"

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors)

    if real_reference:
        real_mean, real_sd = means[0], stds[0]
        ax.axhline(real_mean, color="#F28E2B", linestyle="--", linewidth=2.8)
        ax.axhline(real_mean + real_sd, color="#F28E2B", linestyle=":", linewidth=2.5)
        ax.axhline(real_mean - real_sd, color="#F28E2B", linestyle=":", linewidth=2.5)

    if reference is not None:
        ax.axhline(reference, color="black", linestyle="--", linewidth=2.2)

    ax.set_ylabel(ylabel, fontsize=30)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35, labelsize=25)
    ax.tick_params(axis="y", labelsize=25)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "plots").mkdir(exist_ok=True)
    utility_rows, fidelity_rows, privacy_rows, status_rows = [], [], [], []

    for fold in range(1, 6):
        fold_name = f"fold_{fold:02d}"
        train = pd.read_csv(args.data_root / "folds" / fold_name / "train.csv")
        validation = pd.read_csv(args.data_root / "folds" / fold_name / "validation.csv")
        real_metrics = evaluate_utility(train, validation, args.trees, args.seed + fold)
        utility_rows.append({
            "Fold": fold, "Model": "Real (TRTR)", "Experiment": "TRTR",
            "Training N": len(train), **real_metrics,
        })

        for key in args.models:
            model_name = DISPLAY_NAMES.get(key, key)
            path = args.synthetic_root / key / fold_name / "synthetic.csv"
            if not path.exists():
                status_rows.append({"Fold": fold, "Model": model_name, "Status": "missing", "Reason": str(path)})
                continue
            try:
                synthetic = pd.read_csv(path)
                if synthetic.columns.tolist() != train.columns.tolist() or synthetic.shape != train.shape:
                    raise ValueError(f"invalid shape or columns: {synthetic.shape}")
                utility_rows.append({
                    "Fold": fold, "Model": model_name, "Experiment": "TSTR",
                    "Training N": len(synthetic),
                    **evaluate_utility(synthetic, validation, args.trees, args.seed + fold),
                })
                fidelity_rows.append({"Fold": fold, "Model": model_name, **fidelity_metrics(train, synthetic)})
                privacy_rows.append({"Fold": fold, "Model": model_name, **privacy_metrics(train, validation, synthetic)})
                status_rows.append({"Fold": fold, "Model": model_name, "Status": "success", "Reason": ""})
                print(f"Fold {fold} | {model_name}: SUCCESS")
            except Exception as error:
                status_rows.append({"Fold": fold, "Model": model_name, "Status": "failed", "Reason": str(error)})
                print(f"Fold {fold} | {model_name}: FAILED — {error}")

    utility = pd.DataFrame(utility_rows)
    fidelity = pd.DataFrame(fidelity_rows)
    privacy = pd.DataFrame(privacy_rows)
    status = pd.DataFrame(status_rows)
    utility.to_csv(args.output_root / "utility_fold_results.csv", index=False)
    fidelity.to_csv(args.output_root / "fidelity_fold_results.csv", index=False)
    privacy.to_csv(args.output_root / "privacy_fold_results.csv", index=False)
    status.to_csv(args.output_root / "evaluation_status.csv", index=False)

    utility_metrics = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC"]
    fidelity_metrics_list = ["Mean Continuous KS", "Mean Categorical TVD", "Mean Correlation Difference", "Correlation Structure rho"]
    privacy_metrics_list = ["Exact Real-Synthetic Matches", "Synthetic Duplicates", "Mean DCR", "Median DCR", "5th Percentile DCR", "Mean Real-Real Distance", "DCR Ratio", "MIA AUC", "Mean Member Distance", "Mean Non-member Distance"]
    utility_summary = mean_sd_table(utility, utility_metrics)
    fidelity_summary = mean_sd_table(fidelity, fidelity_metrics_list)
    privacy_summary = mean_sd_table(privacy, privacy_metrics_list)
    utility_summary.to_csv(args.output_root / "utility_summary.csv", index=False)
    fidelity_summary.to_csv(args.output_root / "fidelity_summary.csv", index=False)
    privacy_summary.to_csv(args.output_root / "privacy_summary.csv", index=False)

    trtr = utility[utility["Experiment"] == "TRTR"].set_index("Fold")
    tests = []
    for model, group in utility[utility["Experiment"] == "TSTR"].groupby("Model"):
        group = group.set_index("Fold")
        common = sorted(set(group.index) & set(trtr.index))
        for metric in utility_metrics:
            differences = group.loc[common, metric] - trtr.loc[common, metric]
            try:
                statistic, pvalue = wilcoxon(differences)
            except ValueError:
                statistic, pvalue = np.nan, np.nan
            tests.append({
                "Model": model, "Metric": metric, "N Paired Folds": len(common),
                "Mean Difference TSTR-TRTR": differences.mean(),
                "Wilcoxon Statistic": statistic, "Wilcoxon p-value": pvalue,
            })
    pd.DataFrame(tests).to_csv(args.output_root / "paired_utility_tests.csv", index=False)

    real_auc = float(trtr["Macro ROC-AUC"].mean())
    compact = utility[utility["Experiment"] == "TSTR"].groupby("Model").agg(
        Successful_Folds=("Fold", "count"),
        Utility_ROC_AUC_Mean=("Macro ROC-AUC", "mean"),
        Utility_ROC_AUC_SD=("Macro ROC-AUC", "std"),
        Balanced_Accuracy_Mean=("Balanced Accuracy", "mean"),
        Macro_F1_Mean=("Macro F1", "mean"),
    ).reset_index()
    compact = compact.merge(
        fidelity.groupby("Model").agg(
            Correlation_Difference_Mean=("Mean Correlation Difference", "mean"),
            Correlation_Structure_Rho_Mean=("Correlation Structure rho", "mean"),
            Continuous_KS_Mean=("Mean Continuous KS", "mean"),
            Categorical_TVD_Mean=("Mean Categorical TVD", "mean"),
        ).reset_index(), on="Model", how="left",
    ).merge(
        privacy.groupby("Model").agg(
            DCR_Ratio_Mean=("DCR Ratio", "mean"),
            MIA_AUC_Mean=("MIA AUC", "mean"),
            Exact_Matches_Total=("Exact Real-Synthetic Matches", "sum"),
        ).reset_index(), on="Model", how="left",
    )
    compact.insert(1, "TRTR_ROC_AUC_Reference", real_auc)
    compact.to_csv(args.output_root / "proposal_ready_comparison.csv", index=False)

    save_bar(
        utility_summary,
        "Macro ROC-AUC",
        "Macro ROC-AUC",
        args.output_root / "plots" / "utility_roc_auc.png",
        real_reference=True,
    )
    save_bar(fidelity_summary, "Mean Correlation Difference", "Mean absolute correlation difference", args.output_root / "plots" / "correlation_difference.png")
    save_bar(privacy_summary, "MIA AUC", "Membership-inference AUC", args.output_root / "plots" / "mia_auc.png", reference=0.5)
    save_bar(privacy_summary, "DCR Ratio", "DCR ratio", args.output_root / "plots" / "dcr_ratio.png", reference=1.0)

    run_info = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "models_requested": args.models,
        "classifier": "RandomForestClassifier",
        "trees": args.trees, "class_weight": "balanced",
        "synthetic_resampling_or_balancing": False,
        "privacy_distance": "Gower-style mixed-type distance",
        "correlation": "Spearman on continuous and ordinal columns; nominal columns excluded",
    }
    (args.output_root / "evaluation_metadata.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )
    print("\nProposal-ready comparison:")
    print(compact.round(4).to_string(index=False))
    print("\nResults saved in:", args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
