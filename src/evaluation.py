"""
src/evaluation.py
=================
Model evaluation — metrics, confusion matrices, comparison.

Supports evaluation of:
  - FuzzyClassifier
  - RandomForestModel

Output files:
  results/metrics/model_comparison.csv
  results/figures/fuzzy_confusion_matrix.png
  results/figures/random_forest_confusion_matrix.png
  results/figures/model_comparison.png

Usage:
    from src.evaluation import evaluate_models, save_comparison_csv
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder

from src.config import get_config, resolve_path

logger = logging.getLogger(__name__)

CLASS_NAMES = ["LOW", "MODERATE", "HIGH"]
CLASS_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2}


# ------------------------------------------------------------------ #
# Metrics
# ------------------------------------------------------------------ #

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """Compute classification metrics."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist(),
        "classification_report": classification_report(
            y_true, y_pred,
            target_names=class_names,
            zero_division=0,
        ),
        "n_samples": len(y_true),
    }


# ------------------------------------------------------------------ #
# Cross-validated evaluation for both models
# ------------------------------------------------------------------ #

def evaluate_fuzzy_cv(
    fuzzy_clf,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
    config=None,
) -> Dict:
    """
    Subject-wise CV evaluation for the fuzzy classifier.

    NOTE: The fuzzy classifier is re-trained per fold using only
    the training data for that fold. This ensures fair evaluation.
    """
    from src.fuzzy_classifier import FuzzyClassifier
    cfg = config or get_config()

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_true, all_y_pred = [], []
    fold_accs = []

    logger.info("Fuzzy CV: %d-fold subject-wise evaluation", n_splits)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fold_clf = FuzzyClassifier(config=cfg)
        try:
            fold_clf.train(X_train, y_train, feature_names)
            y_pred = fold_clf.predict(X_test)
        except Exception as e:
            logger.warning("Fuzzy fold %d failed: %s. Using majority class.", fold_idx + 1, e)
            # Fall back to majority class prediction
            majority = np.bincount(y_train).argmax()
            y_pred = np.full_like(y_test, majority)

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        fold_acc = accuracy_score(y_test, y_pred)
        fold_accs.append(fold_acc)
        logger.info("Fuzzy fold %d: acc=%.3f", fold_idx + 1, fold_acc)

    metrics = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    metrics["model_name"] = "Fuzzy Classifier"
    metrics["n_folds"] = n_splits
    metrics["fold_accuracies"] = fold_accs
    return metrics


def evaluate_rf_cv(
    rf_model,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: List[str],
    n_splits: int = 5,
) -> Dict:
    """
    Subject-wise CV evaluation for Random Forest.
    Delegates to rf_model.cross_validate().
    """
    agg = rf_model.cross_validate(X, y, groups, feature_names, n_splits=n_splits)
    # Wrap in standard format
    return {
        "model_name": "Random Forest",
        "accuracy": agg["accuracy_mean"],
        "accuracy_std": agg["accuracy_std"],
        "balanced_accuracy": agg["balanced_accuracy_mean"],
        "precision_macro": agg["precision_macro_mean"],
        "recall_macro": agg["recall_macro_mean"],
        "f1_macro": agg["f1_macro_mean"],
        "f1_macro_std": agg["f1_macro_std"],
        "n_folds": n_splits,
        "confusion_matrix": agg.get("confusion_matrix_all", []),
        "classification_report": agg.get("classification_report", ""),
    }


# ------------------------------------------------------------------ #
# Comparison
# ------------------------------------------------------------------ #

def save_comparison_csv(
    fuzzy_metrics: Dict,
    rf_metrics: Dict,
    config=None,
) -> str:
    """Save model comparison to CSV."""
    cfg = config or get_config()
    out_path = resolve_path(cfg.paths.metrics) / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for m, name in [(fuzzy_metrics, "Fuzzy Classifier"), (rf_metrics, "Random Forest")]:
        rows.append({
            "model": name,
            "accuracy": m.get("accuracy", 0),
            "balanced_accuracy": m.get("balanced_accuracy", 0),
            "precision_macro": m.get("precision_macro", 0),
            "recall_macro": m.get("recall_macro", 0),
            "f1_macro": m.get("f1_macro", 0),
            "n_folds": m.get("n_folds", 5),
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Model comparison saved: %s", out_path)
    return str(out_path)


def load_comparison_csv(config=None) -> Optional[pd.DataFrame]:
    """Load model comparison CSV if it exists."""
    cfg = config or get_config()
    path = resolve_path(cfg.paths.metrics) / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ------------------------------------------------------------------ #
# Confusion matrix plot
# ------------------------------------------------------------------ #

def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    class_names: List[str] = CLASS_NAMES,
    config=None,
) -> str:
    """Save a styled confusion matrix figure."""
    cfg = config or get_config()
    figures_dir = resolve_path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)

    filename = model_name.lower().replace(" ", "_") + "_confusion_matrix.png"
    out_path = figures_dir / filename

    # Normalize
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = cm_norm / row_sums

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        linewidths=0.5,
        linecolor="gray",
        vmin=0,
        vmax=1,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix\n(row-normalised)", fontsize=13, pad=10)

    # Add raw counts
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j + 0.5, i + 0.75,
                f"(n={cm[i,j]})",
                ha="center", va="center",
                fontsize=8, color="gray",
            )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved: %s", out_path)
    return str(out_path)


def plot_model_comparison(
    fuzzy_metrics: Dict,
    rf_metrics: Dict,
    config=None,
) -> str:
    """Save a bar chart comparing model metrics."""
    cfg = config or get_config()
    figures_dir = resolve_path(cfg.paths.figures)
    figures_dir.mkdir(parents=True, exist_ok=True)
    out_path = figures_dir / "model_comparison.png"

    metric_names = ["accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "f1_macro"]
    labels = ["Accuracy", "Balanced\nAccuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"]

    fuzzy_vals = [fuzzy_metrics.get(m, 0) for m in metric_names]
    rf_vals = [rf_metrics.get(m, 0) for m in metric_names]

    x = np.arange(len(metric_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, fuzzy_vals, width, label="Fuzzy Classifier",
                   color="#4C72B0", alpha=0.85, edgecolor="white")
    bars2 = ax.bar(x + width/2, rf_vals, width, label="Random Forest",
                   color="#DD8452", alpha=0.85, edgecolor="white")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison — Cross-validated Performance\n(Subject-wise CV)", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(
            f"{h:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison plot saved: %s", out_path)
    return str(out_path)
