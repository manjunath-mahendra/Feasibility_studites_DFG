#!/usr/bin/env python3
"""Compare real-, NextConvGeN-, and two-source ensemble classifiers.

Each outer fold uses its existing real train/validation split and matching
NextConvGeN synthetic training data. The outer validation fold is used only
once for final evaluation. Weighted voting and stacking are learned with
inner cross-validation on the outer training data.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier


TARGET = "diagnosis"
CLASSES = np.array([1, 2, 3], dtype=int)
NOMINAL = ["sex", "bariatricSurgery"]
ORDINAL = [
    "DiabetesMellitusII", "ArterielleHypertonie", "Hyperlipidämie",
    "Hypertriglycerinämie", "KHK",
]
CATEGORICAL = NOMINAL + ORDINAL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data/xconvgen"))
    parser.add_argument("--output-root", type=Path, default=Path("ensemble_results"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def make_preprocessor(columns: list[str]) -> ColumnTransformer:
    categorical = [column for column in CATEGORICAL if column in columns]
    continuous = [column for column in columns if column not in categorical]
    return ColumnTransformer([
        ("continuous", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), continuous),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])


def make_model(kind: str, columns: list[str], trees: int, seed: int) -> Pipeline:
    if kind == "rf":
        classifier = RandomForestClassifier(
            n_estimators=trees,
            class_weight="balanced",
            max_features="sqrt",
            random_state=seed,
            n_jobs=-1,
        )
    elif kind == "bagging":
        classifier = BaggingClassifier(
            estimator=DecisionTreeClassifier(
                class_weight="balanced", max_features="sqrt", random_state=seed
            ),
            n_estimators=trees,
            random_state=seed,
            n_jobs=-1,
        )
    elif kind == "boosting":
        classifier = AdaBoostClassifier(
            n_estimators=min(trees, 300), learning_rate=0.05, random_state=seed
        )
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    return Pipeline([
        ("preprocessor", make_preprocessor(columns)),
        ("classifier", classifier),
    ])


def aligned_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    fitted_classes = model.named_steps["classifier"].classes_.astype(int)
    aligned = np.zeros((len(x), len(CLASSES)), dtype=float)
    for source_index, label in enumerate(fitted_classes):
        matches = np.where(CLASSES == label)[0]
        if len(matches):
            aligned[:, matches[0]] = raw[:, source_index]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return np.divide(aligned, row_sums, out=np.zeros_like(aligned), where=row_sums != 0)


def macro_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    binary = label_binarize(y_true, classes=CLASSES)
    return float(roc_auc_score(binary, probability, average="macro", multi_class="ovr"))


def calculate_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    prediction = CLASSES[np.argmax(probability, axis=1)]
    return {
        "Accuracy": float(accuracy_score(y_true, prediction)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "Macro F1": float(f1_score(y_true, prediction, average="macro", zero_division=0)),
        "Weighted F1": float(f1_score(y_true, prediction, average="weighted", zero_division=0)),
        "Macro ROC-AUC": macro_auc(y_true, probability),
    }


def inner_oof_probabilities(
    x_real: pd.DataFrame,
    y_real: np.ndarray,
    x_syn: pd.DataFrame,
    y_syn: np.ndarray,
    trees: int,
    requested_splits: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    minimum_class_count = min(np.bincount(y_real)[1:].min(), np.bincount(y_syn)[1:].min())
    n_splits = min(requested_splits, int(minimum_class_count))
    if n_splits < 2:
        raise ValueError("At least two observations per class are required for inner CV")

    real_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    syn_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + 10_000)
    real_splits = list(real_cv.split(x_real, y_real))
    syn_splits = list(syn_cv.split(x_syn, y_syn))
    real_oof = np.zeros((len(x_real), len(CLASSES)), dtype=float)
    syn_oof = np.zeros_like(real_oof)

    for inner_fold, ((real_train, real_valid), (syn_train, _)) in enumerate(
        zip(real_splits, syn_splits), start=1
    ):
        real_model = make_model("rf", list(x_real.columns), trees, seed + inner_fold)
        syn_model = make_model("rf", list(x_syn.columns), trees, seed + 100 + inner_fold)
        real_model.fit(x_real.iloc[real_train], y_real[real_train])
        syn_model.fit(x_syn.iloc[syn_train], y_syn[syn_train])
        real_oof[real_valid] = aligned_probability(real_model, x_real.iloc[real_valid])
        syn_oof[real_valid] = aligned_probability(syn_model, x_real.iloc[real_valid])
    return real_oof, syn_oof


def select_real_weight(y: np.ndarray, real_oof: np.ndarray, syn_oof: np.ndarray) -> tuple[float, float]:
    best_weight, best_score = 0.5, -np.inf
    for weight in np.linspace(0.0, 1.0, 21):
        score = macro_auc(y, weight * real_oof + (1.0 - weight) * syn_oof)
        if score > best_score:
            best_weight, best_score = float(weight), float(score)
    return best_weight, best_score


def fit_source_pair(
    kind: str,
    x_real: pd.DataFrame,
    y_real: np.ndarray,
    x_syn: pd.DataFrame,
    y_syn: np.ndarray,
    x_valid: pd.DataFrame,
    trees: int,
    seed: int,
    same_seed: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    real_model = make_model(kind, list(x_real.columns), trees, seed)
    # The benchmark evaluates the real- and synthetic-trained RFs with the
    # same fold-specific random state. Preserve that behavior for the two
    # reported baselines so their values reproduce exactly.
    syn_seed = seed if same_seed else seed + 1_000
    syn_model = make_model(kind, list(x_syn.columns), trees, syn_seed)
    real_model.fit(x_real, y_real)
    syn_model.fit(x_syn, y_syn)
    return aligned_probability(real_model, x_valid), aligned_probability(syn_model, x_valid)


def load_fold(args: argparse.Namespace, fold: int):
    fold_name = f"fold_{fold:02d}"
    train_path = args.data_root / "folds" / fold_name / "train.csv"
    validation_path = args.data_root / "folds" / fold_name / "validation.csv"
    if not validation_path.exists():
        validation_path = args.data_root / "folds" / fold_name / "test.csv"
    synthetic_path = args.synthetic_root / fold_name / "synthetic.csv"
    for path in [train_path, validation_path, synthetic_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing required file: {path}")
    train = pd.read_csv(train_path)
    validation = pd.read_csv(validation_path)
    synthetic = pd.read_csv(synthetic_path)
    if train.columns.tolist() != validation.columns.tolist() or train.columns.tolist() != synthetic.columns.tolist():
        raise ValueError(f"Column mismatch in {fold_name}")
    return fold_name, train, validation, synthetic


def save_plot(summary: pd.DataFrame, output_path: Path) -> None:
    labels = summary["Method"].tolist()
    means = summary["Macro ROC-AUC Mean"].to_numpy()
    stds = summary["Macro ROC-AUC SD"].fillna(0).to_numpy()
    colors = ["#F28E2B" if label == "Real only" else "#4C78A8" for label in labels]
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    ax.set_ylabel("Macro ROC-AUC", fontsize=30)
    ax.tick_params(axis="x", rotation=35, labelsize=21)
    ax.tick_params(axis="y", labelsize=25)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metric_rows, prediction_rows, weight_rows = [], [], []

    for fold in args.folds:
        fold_name, train, validation, synthetic = load_fold(args, fold)
        feature_columns = [column for column in train.columns if column != TARGET]
        x_real, y_real = train[feature_columns], train[TARGET].astype(int).to_numpy()
        x_syn, y_syn = synthetic[feature_columns], synthetic[TARGET].astype(int).to_numpy()
        x_valid, y_valid = validation[feature_columns], validation[TARGET].astype(int).to_numpy()

        real_oof, syn_oof = inner_oof_probabilities(
            x_real, y_real, x_syn, y_syn, args.trees, args.inner_folds, args.seed + fold * 100
        )
        real_weight, inner_auc = select_real_weight(y_real, real_oof, syn_oof)
        meta_model = LogisticRegression(
            class_weight="balanced", max_iter=2_000, random_state=args.seed + fold
        )
        meta_model.fit(np.hstack([real_oof, syn_oof]), y_real)

        real_rf, syn_rf = fit_source_pair(
            "rf", x_real, y_real, x_syn, y_syn, x_valid,
            args.trees, args.seed + fold, same_seed=True,
        )
        bag_real, bag_syn = fit_source_pair(
            "bagging", x_real, y_real, x_syn, y_syn, x_valid,
            args.trees, args.seed + fold * 2_000,
        )
        boost_real, boost_syn = fit_source_pair(
            "boosting", x_real, y_real, x_syn, y_syn, x_valid,
            args.trees, args.seed + fold * 3_000,
        )

        probabilities = {
            "Real only": real_rf,
            "NextConvGeN only": syn_rf,
            "Soft voting": 0.5 * real_rf + 0.5 * syn_rf,
            "Weighted voting": real_weight * real_rf + (1.0 - real_weight) * syn_rf,
            "Stacking": aligned_probability_from_meta(
                meta_model, np.hstack([real_rf, syn_rf])
            ),
            "Bagging": 0.5 * bag_real + 0.5 * bag_syn,
            "Boosting": 0.5 * boost_real + 0.5 * boost_syn,
        }

        weight_rows.append({
            "Fold": fold,
            "Real Weight": real_weight,
            "NextConvGeN Weight": 1.0 - real_weight,
            "Inner-CV Macro ROC-AUC": inner_auc,
        })
        for method, probability in probabilities.items():
            metric_rows.append({"Fold": fold, "Method": method, **calculate_metrics(y_valid, probability)})
            predictions = CLASSES[np.argmax(probability, axis=1)]
            for row_index, (truth, prediction, values) in enumerate(zip(y_valid, predictions, probability)):
                prediction_rows.append({
                    "Fold": fold, "Row": row_index, "Method": method,
                    "True Diagnosis": int(truth), "Predicted Diagnosis": int(prediction),
                    "Probability Class 1": values[0], "Probability Class 2": values[1],
                    "Probability Class 3": values[2],
                })
        print(f"{fold_name}: real weight={real_weight:.2f}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_root / "foldwise_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_root / "holdout_predictions.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(args.output_root / "selected_voting_weights.csv", index=False)

    metric_columns = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC"]
    summary_parts = []
    for metric in metric_columns:
        part = metrics.groupby("Method", sort=False)[metric].agg(["mean", "std"]).reset_index()
        part = part.rename(columns={"mean": f"{metric} Mean", "std": f"{metric} SD"})
        summary_parts.append(part)
    summary = summary_parts[0]
    for part in summary_parts[1:]:
        summary = summary.merge(part, on="Method", how="left")
    summary.to_csv(args.output_root / "mean_sd_summary.csv", index=False)
    save_plot(summary, args.output_root / "macro_roc_auc_comparison.png")

    metadata = {
        "outer_evaluation": "Untouched real validation fold",
        "base_classifier": "RandomForestClassifier",
        "baseline_seed_policy": "Matches evaluate_all_generators.py: random_state = seed + outer fold for both real-only and NextConvGeN-only models",
        "soft_voting": "Equal average of real- and NextConvGeN-trained probabilities",
        "weighted_voting": "Real-source weight selected by inner-CV macro ROC-AUC",
        "stacking": "Multinomial logistic regression fitted to inner out-of-fold probabilities",
        "bagging": "Average of real- and NextConvGeN-trained BaggingClassifier probabilities",
        "boosting": "Average of real- and NextConvGeN-trained AdaBoostClassifier probabilities",
        "folds": args.folds,
        "seed": args.seed,
    }
    with open(args.output_root / "methodology.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nMean ± SD across outer folds:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nResults saved in: {args.output_root.resolve()}")
    return 0


def aligned_probability_from_meta(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(features)
    aligned = np.zeros((len(features), len(CLASSES)), dtype=float)
    for source_index, label in enumerate(model.classes_.astype(int)):
        matches = np.where(CLASSES == label)[0]
        if len(matches):
            aligned[:, matches[0]] = raw[:, source_index]
    return aligned


if __name__ == "__main__":
    raise SystemExit(main())
