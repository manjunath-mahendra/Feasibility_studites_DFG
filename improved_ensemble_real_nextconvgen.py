#!/usr/bin/env python3
"""Nested-CV improvements for real + NextConvGeN classification.

Requires ensemble_real_nextconvgen.py in the same directory. All model and
decision-rule selection uses only the outer training fold. Final evaluation is
performed once on the untouched real validation fold.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import ensemble_real_nextconvgen as base


RATIOS = [0.25, 0.50, 1.00]
RF_CONFIGS = [
    {"max_depth": 3, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"max_depth": 5, "min_samples_leaf": 2, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 4, "max_features": "sqrt"},
    {"max_depth": None, "min_samples_leaf": 2, "max_features": 0.5},
]
STACKING_CONFIGS = [
    {"C": 0.01, "class_weight": "balanced"},
    {"C": 0.10, "class_weight": "balanced"},
    {"C": 1.00, "class_weight": "balanced"},
    {"C": 10.0, "class_weight": "balanced"},
    {"C": 0.10, "class_weight": None},
    {"C": 1.00, "class_weight": None},
]
CLASS_MULTIPLIERS = [0.75, 1.0, 1.25, 1.5, 2.0]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("fibrosis_benchmark_data"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("synthetic_data/xconvgen"))
    parser.add_argument("--output-root", type=Path, default=Path("improved_ensemble_results"))
    parser.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--trees", type=int, default=500)
    parser.add_argument("--tuning-trees", type=int, default=200)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_synthetic(x, y, ratio, seed):
    """Take a reproducible class-stratified fraction of synthetic rows."""
    rng = np.random.default_rng(seed)
    selected = []
    for label in base.CLASSES:
        candidates = np.flatnonzero(y == label)
        count = max(1, int(round(len(candidates) * ratio)))
        count = min(count, len(candidates))
        selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
    selected = np.asarray(selected)
    rng.shuffle(selected)
    return x.iloc[selected].reset_index(drop=True), y[selected]


def make_tuned_rf(columns, trees, seed, config):
    return Pipeline([
        ("preprocessor", base.make_preprocessor(columns)),
        ("classifier", RandomForestClassifier(
            n_estimators=trees,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
            **config,
        )),
    ])


def combine(x_real, y_real, x_syn, y_syn):
    x = pd.concat([x_real.reset_index(drop=True), x_syn.reset_index(drop=True)], ignore_index=True)
    y = np.concatenate([y_real, y_syn])
    return x, y


def choose_augmented_model(x_real, y_real, x_syn, y_syn, args, fold):
    """Jointly select augmentation ratio and RF regularization by inner CV."""
    n_splits = min(args.inner_folds, int(np.unique(y_real, return_counts=True)[1].min()))
    real_cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=args.seed + fold * 100).split(x_real, y_real))
    syn_cv = list(StratifiedKFold(n_splits=n_splits, shuffle=True,
                                 random_state=args.seed + fold * 100 + 1).split(x_syn, y_syn))
    records = []
    for ratio, config_id in itertools.product(RATIOS, range(len(RF_CONFIGS))):
        fold_scores = []
        for inner, ((r_train, r_valid), (s_train, _)) in enumerate(zip(real_cv, syn_cv), start=1):
            sx, sy = sample_synthetic(
                x_syn.iloc[s_train], y_syn[s_train], ratio,
                args.seed + fold * 10_000 + config_id * 100 + inner,
            )
            train_x, train_y = combine(x_real.iloc[r_train], y_real[r_train], sx, sy)
            model = make_tuned_rf(
                list(x_real.columns), args.tuning_trees,
                args.seed + fold * 1_000 + inner, RF_CONFIGS[config_id],
            )
            model.fit(train_x, train_y)
            probability = base.aligned_probability(model, x_real.iloc[r_valid])
            metrics = base.calculate_metrics(y_real[r_valid], probability)
            fold_scores.append(metrics)
        records.append({
            "ratio": ratio,
            "config_id": config_id,
            "macro_f1": np.mean([x["Macro F1"] for x in fold_scores]),
            "balanced_accuracy": np.mean([x["Balanced Accuracy"] for x in fold_scores]),
            "macro_auc": np.mean([x["Macro ROC-AUC"] for x in fold_scores]),
        })
    # Primary selection criterion: macro F1; use balanced accuracy and AUROC as tie-breakers.
    winner = max(records, key=lambda x: (x["macro_f1"], x["balanced_accuracy"], x["macro_auc"]))
    return winner, records


def meta_oof_predictions(features, labels, config, n_splits, seed):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probability = np.zeros((len(labels), len(base.CLASSES)))
    for train_idx, valid_idx in cv.split(features, labels):
        model = LogisticRegression(max_iter=2_000, random_state=seed, **config)
        model.fit(features[train_idx], labels[train_idx])
        raw = model.predict_proba(features[valid_idx])
        for source, label in enumerate(model.classes_.astype(int)):
            probability[valid_idx, np.where(base.CLASSES == label)[0][0]] = raw[:, source]
    return probability


def adjusted_probability(probability, multipliers):
    adjusted = probability * np.asarray(multipliers)[None, :]
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def choose_stacking(x_real, y_real, x_syn, y_syn, args, fold):
    real_oof, syn_oof = base.inner_oof_probabilities(
        x_real, y_real, x_syn, y_syn, args.tuning_trees,
        args.inner_folds, args.seed + fold * 100,
    )
    features = np.hstack([real_oof, syn_oof])
    n_splits = min(args.inner_folds, int(np.unique(y_real, return_counts=True)[1].min()))
    candidates = []
    for config_id, config in enumerate(STACKING_CONFIGS):
        meta_oof = meta_oof_predictions(features, y_real, config, n_splits,
                                        args.seed + fold * 2_000 + config_id)
        metrics = base.calculate_metrics(y_real, meta_oof)
        candidates.append((metrics["Macro F1"], metrics["Balanced Accuracy"],
                           metrics["Macro ROC-AUC"], config_id, meta_oof))
    _, _, _, config_id, best_meta_oof = max(candidates, key=lambda x: x[:3])

    best_multipliers, best_score = (1.0, 1.0, 1.0), -np.inf
    # Keep class 1 as the reference and tune only minority-class decision weights.
    for class_2, class_3 in itertools.product(CLASS_MULTIPLIERS, repeat=2):
        multipliers = (1.0, class_2, class_3)
        prediction = base.CLASSES[np.argmax(adjusted_probability(best_meta_oof, multipliers), axis=1)]
        score = f1_score(y_real, prediction, average="macro", zero_division=0)
        if score > best_score:
            best_score, best_multipliers = score, multipliers
    return STACKING_CONFIGS[config_id], best_multipliers, real_oof, syn_oof, best_score


def plot_summary(summary, path):
    labels = summary["Method"].tolist()
    means = summary["Macro ROC-AUC Mean"].to_numpy()
    stds = summary["Macro ROC-AUC SD"].fillna(0).to_numpy()
    colors = ["#F28E2B" if label == "Real only" else "#4C78A8" for label in labels]
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.bar(labels, means, yerr=stds, capsize=6, color=colors)
    ax.set_ylabel("Macro ROC-AUC", fontsize=30)
    ax.tick_params(axis="x", rotation=35, labelsize=20)
    ax.tick_params(axis="y", labelsize=25)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    metric_rows, selection_rows, prediction_rows = [], [], []

    for fold in args.folds:
        fold_name, train, validation, synthetic = base.load_fold(args, fold)
        columns = [c for c in train.columns if c != base.TARGET]
        xr, yr = train[columns], train[base.TARGET].astype(int).to_numpy()
        xs, ys = synthetic[columns], synthetic[base.TARGET].astype(int).to_numpy()
        xv, yv = validation[columns], validation[base.TARGET].astype(int).to_numpy()

        # Exact benchmark baselines.
        real_p, syn_p = base.fit_source_pair(
            "rf", xr, yr, xs, ys, xv, args.trees, args.seed + fold, same_seed=True
        )

        # Inner-CV-selected real + synthetic augmentation.
        aug_choice, _ = choose_augmented_model(xr, yr, xs, ys, args, fold)
        sx, sy = sample_synthetic(xs, ys, aug_choice["ratio"], args.seed + fold * 50_000)
        aug_x, aug_y = combine(xr, yr, sx, sy)
        aug_model = make_tuned_rf(columns, args.trees, args.seed + fold,
                                  RF_CONFIGS[aug_choice["config_id"]])
        aug_model.fit(aug_x, aug_y)
        aug_p = base.aligned_probability(aug_model, xv)

        # Regularized stacking and inner-CV class-decision adjustment.
        stack_config, multipliers, real_oof, syn_oof, inner_stack_f1 = choose_stacking(
            xr, yr, xs, ys, args, fold
        )
        meta = LogisticRegression(max_iter=2_000, random_state=args.seed + fold, **stack_config)
        meta.fit(np.hstack([real_oof, syn_oof]), yr)
        stack_p = base.aligned_probability_from_meta(meta, np.hstack([real_p, syn_p]))
        adjusted_stack_p = adjusted_probability(stack_p, multipliers)

        methods = {
            "Real only": real_p,
            "NextConvGeN only": syn_p,
            "Selected augmentation": aug_p,
            "Tuned stacking": stack_p,
            "Tuned stacking + decision adjustment": adjusted_stack_p,
        }
        selection_rows.append({
            "Fold": fold,
            "Synthetic Ratio": aug_choice["ratio"],
            "RF Configuration": json.dumps(RF_CONFIGS[aug_choice["config_id"]]),
            "Augmentation Inner Macro F1": aug_choice["macro_f1"],
            "Stacking Configuration": json.dumps(stack_config),
            "Class Multipliers": json.dumps(multipliers),
            "Stacking Inner Macro F1": inner_stack_f1,
        })
        for method, probability in methods.items():
            metric_rows.append({"Fold": fold, "Method": method,
                                **base.calculate_metrics(yv, probability)})
            prediction = base.CLASSES[np.argmax(probability, axis=1)]
            for row, (truth, pred, prob) in enumerate(zip(yv, prediction, probability)):
                prediction_rows.append({
                    "Fold": fold, "Row": row, "Method": method,
                    "True Diagnosis": int(truth), "Predicted Diagnosis": int(pred),
                    "Probability Class 1": prob[0], "Probability Class 2": prob[1],
                    "Probability Class 3": prob[2],
                })
        print(f"{fold_name}: ratio={aug_choice['ratio']:.2f}, "
              f"RF config={aug_choice['config_id']}, stack C={stack_config['C']}, "
              f"multipliers={multipliers}")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(args.output_root / "foldwise_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(args.output_root / "inner_cv_selections.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(args.output_root / "holdout_predictions.csv", index=False)

    metric_names = ["Accuracy", "Balanced Accuracy", "Macro F1", "Weighted F1", "Macro ROC-AUC"]
    summary = None
    for metric in metric_names:
        part = metrics.groupby("Method", sort=False)[metric].agg(["mean", "std"]).reset_index()
        part = part.rename(columns={"mean": f"{metric} Mean", "std": f"{metric} SD"})
        summary = part if summary is None else summary.merge(part, on="Method", how="left")
    summary.to_csv(args.output_root / "mean_sd_summary.csv", index=False)
    plot_summary(summary, args.output_root / "macro_roc_auc_comparison.png")
    print("\nMean ± SD across outer folds:")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nResults saved in: {args.output_root.resolve()}")


if __name__ == "__main__":
    main()
